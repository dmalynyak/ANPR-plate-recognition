import os
import torch
from src.data_loaders.yolov2_dataload import garbage_names_clean, load_one_image_path

# structure:
# val_map:
    # input: model, anchors(5,2), img_dir, label_dir, device
    # output: dict{3} (mAP_50, mAP_75, mAP_5095)

    # get_all_boxes: ---------------------------------- start -------------------------------------------
        # input: model, anchors(5,2), img_dir, label_dir, device, batch_size, max_det
#GPU    # output: dets(Total_det, 7), gts(Total_gt, 6)
        # output info: makes list dets, gts and concatenates to tensor at the very end (ONCE)

        # per batch:
            # dataload.load_one_image: (per one image in batch)
                # input: img_path
#GPU            # output: (1, 3, 416, 416) raw img tensor from dataset
                # runs on every img in batch
                # cat all img tensors in batch. returns (B, 3, 416, 416)

            # forward pass: (runs for whole batch at once)
#GPU            # input: (B, 3, 416, 416) from dataload.load_one_image
                # ouput: (B, 13, 13, 5, 25) model forward pass output

            # get_detected_boxes: (runs per image in batch, pred = preds[b])
#GPU            # input: pred(13, 13, 5, 25), anchors(5,2), conf_threshold, iou_boxes_threshold
                # output: (N, 6) kept boxes after NMS. row = [x1, y1, x2, y2, score, cls_id]

                    # decode_prediction:
#GPU                    # input: pred(13, 13, 5, 25), anchors(5,2), device
                        # output: dec(13, 13, 5, 25) sigmoid/exp/softmax + cell & anchor offsets applied

                        # obj*cls_score -> score(13, 13, 5)
#GPU                    # corners x1y1x2y2, stack+flatten -> dets(845, 6), 845 = 13*13*5
                        # conf_mask (score > conf_threshold) -> (K, 6) - for one image before NMS

                        # nms: - resolves overlapping of detections (non-max-suppresion)
#GPU                        # input: detections(K, 6), iou_boxes_threshold
                            # output: (N, 6) one box kept per overlapping group, per class

            # max_det capture + adding img_id:
#GPU            # input: (N, 6) from get_detected_boxes after NMS
                # output: (n, 7), n <= max_det. row = [img_id, x1, y1, x2, y2, score, cls]

# ============== (n, 7) GPU -> CPU transfer ==================

            # get_target_boxes:
#CPU            # input: label_path
                # output: (M, 5) GT boxes in grid units. row = [x1, y1, x2, y2, cls]
            # add img_id -> (M, 6)

        # after all batches: detected_boxes = [Total_gt (7, ) tensors], target_boxes = [...] then CAT ONCE
#CPU        # dets(Total_det, 7), gts(Total_gt, 6) <- output
    # get_all_boxes: ---------------------------------- end -------------------------------------------


    # mean_average_precision:
        # input: dets(Total_det, 7), gts(Total_gt, 6), num_cls=20
#CPU    # output: dict{3} (mAP_50, mAP_75, mAP_5095)

        # present_classes = gts[:,5].unique() # skips classes with no GT

        # per present class:
#CPU        # dets_c(K, 7), gts_c(T, 6) # detections/GTs with this cls_id

            # class_match_ious: (runs ONCE per class)
                # input: dets_c(K, 7), gts_c(T, 6)
#CPU            # output: best_iou(K,), best_gt(K,)   # score-sorted
                # IoU with ^ closes GT=[0,1].  ^ row index in gts_c of GT object that best_iou[i] has.
                # uses iou_thrd_mask(det(4,), gts(g, 4)) -> (g,) per detection

            # per threshold t=0.50, 0.55,..., 0.95 (x10):
#CPU            # ap_from_matches:
                    # input: best_iou(K,), best_gt(K,), num_gt (scalar), t (scalar)
                    # output: ap (scalar)

                    # every_point_interpolation:
#CPU                    # input: recall(K,), precision(K,)
                        # output: ap (scalar) area under PR curve

            # ap_table(num_present, 10)

        # ap_table[:,0].mean / [:,5].mean / all.mean -> dict{3}


def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
            # with torch.autocast(device_type=device.type, dtype=amp_dtype):
            predictions = model(images)
            loss = criterion(predictions, targets)
            total += loss.item()
    return total / len(loader)


def val_map(model, anchors, img_dir_path, label_dir_path, device, conf_threshold=0.001, iou_boxes_threshold=0.5,
            batch_size=16, max_det=100):
    dets, gts = get_all_boxes(model, anchors, img_dir_path, label_dir_path, device, conf_threshold, iou_boxes_threshold,
                              batch_size, max_det)

    mAP = mean_average_precision(dets, gts)

    print("mAP@0.5: {:.4f}  mAP@0.75: {:.4f}  mAP@[.5:.95]: {:.4f}".format(
        mAP["mAP_50"], mAP["mAP_75"], mAP["mAP_5095"]))
    return mAP


# gives final inference answer (before postprocessing)
# pred: (13, 13, 5, 25) decoded: (13, 13, 5, 25) FOR ONE IMAGE
def decode_prediction(pred, anchors, device):
    with torch.no_grad():
        anchors = anchors.to(device)
        dec = torch.clone(pred)  # makes independant copy
        S = 13

        cx = torch.arange(S, device=device).view(1, S, 1) # (1, 13, 1) broadcasts to (13, 13, 5) where (13, i, 5) = i
        cy = torch.arange(S, device=device).view(S, 1, 1) # (13, 1, 1)
        pw = anchors[:, 0].view(1, 1, 5) # broadcasts to (13, 13, 5)
        ph = anchors[:, 1].view(1, 1, 5) # broadcasts to (13, 13, 5)

        dec[..., 0] = torch.sigmoid(pred[..., 0]) + cx
        dec[..., 1] = torch.sigmoid(pred[..., 1]) + cy
        dec[..., 2] = pw * torch.exp(pred[..., 2])
        dec[..., 3] = ph * torch.exp(pred[..., 3])
        dec[..., 4] = torch.sigmoid(pred[..., 4])
        dec[..., 5:25] = torch.softmax(pred[..., 5:25], dim=-1)
        return dec # xy - img units, wh - cell units


def iou_thrd_mask(box, boxes): # (4,) and (N, 4) returns (N,)
    x1 = torch.max(box[0], boxes[:, 0]) # (M,)
    y1 = torch.max(box[1], boxes[:, 1]) # (M,)
    x2 = torch.min(box[2], boxes[:, 2]) # (M,)
    y2 = torch.min(box[3], boxes[:, 3]) # (M,)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) # (M,)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (M,)
    union = area_box + area_boxes - inter
    return inter / (union + 1e-9)


def nms(detections, iou_boxes_threshold=0.5):
    keep_det = []
    for cls in detections[:, 5].unique():  # loop (0-20) existing object classes
        cls_det = detections[detections[:, 5] == cls]
        order = cls_det[:, 4].argsort(descending=True) # finds descending order of scores in those detections
        cls_det = cls_det[order] # rearanges class detection in right order

        while cls_det.shape[0] > 0:
            keep_det.append(cls_det[0, :])
            if cls_det.shape[0] == 1:
                break
            iou = iou_thrd_mask(cls_det[0, :4], cls_det[1:, :4])
            save_mask = iou < iou_boxes_threshold
            cls_det = cls_det[1:, :][save_mask, :]

    if len(keep_det) == 0:
        return detections.new_zeros((0, 6))
    return torch.stack(keep_det, dim=0)


# per one image
def get_detected_boxes(pred, anchors, conf_threshold=0.001, iou_boxes_threshold=0.5):
    device = pred.device
    anchors = anchors.to(device)
    dec = decode_prediction(pred, anchors, device)

    obj = dec[..., 4]  # (13, 13, 5)
    cls_probs = dec[..., 5:25] # (13, 13, 5, 20)
    cls_score, cls_id = cls_probs.max(dim=-1) # both (13, 13, 5)
    score = obj * cls_score # (13, 13, 5)

    x1 = dec[..., 0] - dec[..., 2] / 2
    y1 = dec[..., 1] - dec[..., 3] / 2
    x2 = dec[..., 0] + dec[..., 2] / 2
    y2 = dec[..., 1] + dec[..., 3] / 2

    dets = torch.stack([    # 6 tensors (13, 13, 5) get flattened to (845,)
        x1.reshape(-1),     # 6 tensors (845, ) get stacked to (845, 6)
        y1.reshape(-1),             # [x1[0], y1[0], x2[0], y2[0], score[0], class_id[0]]
        x2.reshape(-1),             # [x1[1], y1[1], x2[1], y2[1], score[1], class_id[1]]
        y2.reshape(-1),             # ...
        score.reshape(-1),          # [x1[844], y1[844], x2[844], y2[844], score[844], class_id[844]]
        cls_id.reshape(-1).float()],
        dim = -1)

    conf_mask = (score > conf_threshold).reshape(-1)
    detections = dets[conf_mask]
    detected_boxes = nms(detections, iou_boxes_threshold)
    return detected_boxes  # (N, 6)


def get_target_boxes(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            if not line.split():
                continue
            cls, cx, cy, w, h = map(float, line.split())
            cx, cy, w, h = cx * 13, cy * 13, w * 13, h * 13
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2, cls])

    if len(boxes) == 0:
        return torch.zeros((0, 5))
    return torch.tensor(boxes, dtype=torch.float32)


@torch.no_grad()
def get_all_boxes(model, anchors, img_dir_path, label_dir_path, device,
                  conf_threshold=0.001, iou_boxes_threshold=0.5,
                  batch_size=16, max_det=100):
    model.eval()
    anchors = anchors.to(device)

    all_names = sorted(os.listdir(img_dir_path))
    good_names = garbage_names_clean(all_names, label_dir_path)

    detected_boxes = [] # (Total, N, 7) [img_id, x1[0], y1[0], x2[0], y2[0], score[0], class_id[0]]
    target_boxes = [] # (Total, M, 5) [img_id, x1, y1, x2, y2]

    for start in range(0, len(good_names), batch_size):
        batch_names = good_names[start:start + batch_size]

        # now runs for batch and not for single image
        imgs = torch.cat(
            [load_one_image_path(os.path.join(img_dir_path, n), 416, device)
             for n in batch_names], dim=0)

        amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        # with torch.autocast(device_type=device.type, dtype=amp_dtype):
        preds = model(imgs)
        preds = preds.float()

        for b, name in enumerate(batch_names):
            img_id = start + b


            one_det = get_detected_boxes(preds[b], anchors,
                                         conf_threshold, iou_boxes_threshold)

            if one_det.shape[0] > max_det:
                keep = one_det[:, 4].argsort(descending=True)[:max_det]
                one_det = one_det[keep]
            one_det = one_det.to("cpu")

            idx = torch.full((one_det.shape[0], 1), float(img_id))
            detected_boxes.append(torch.cat([idx, one_det], dim=1))

            label_path = os.path.join(label_dir_path,
                                      os.path.splitext(name)[0] + '.txt')
            one_gt = get_target_boxes(label_path) # (M,5) CPU
            idx = torch.full((one_gt.shape[0], 1), float(img_id))
            target_boxes.append(torch.cat([idx, one_gt], dim=1))

    dets = torch.cat(detected_boxes, dim=0) if detected_boxes else torch.zeros((0, 7))
    gts = torch.cat(target_boxes, dim=0) if target_boxes else torch.zeros((0, 6))
    return dets, gts


def class_match_ious(dets_c, gts_c):
    order = dets_c[:, 5].argsort(descending=True)   # sort by score desc
    dets_c = dets_c[order]

    K = dets_c.shape[0]
    best_iou = torch.zeros(K)
    best_gt = torch.full((K,), -1, dtype=torch.long)  # global row in gts_c

    for i in range(K):
        img_id = dets_c[i, 0]
        gt_mask = gts_c[:, 0] == img_id  # mask of real box in image det
        gt_idx = gt_mask.nonzero(as_tuple=True)[0] # rows in gts_c / matched
        if gt_idx.numel() == 0:  # no GT here -> false positive
            continue
        # IoU of this detection against each GT box in the image (vectorized)
        ious = iou_thrd_mask(dets_c[i, 1:5], gts_c[gt_mask, 1:5])
        biou, bj = ious.max(dim=0)
        best_iou[i] = biou   # best-overlapping GT + its index
        best_gt[i] = gt_idx[bj]   # best-overlapping GT + its index
    return best_iou, best_gt



def ap_from_matches(best_iou, best_gt, num_gt, iou_threshold):
    K = best_iou.shape[0]
    if num_gt == 0:
        return None
    if K == 0:
        return 0.0

    matched = torch.zeros(num_gt, dtype=torch.bool)
    TP = torch.zeros(K)
    FP = torch.zeros(K)

    for i in range(K):
        gt = int(best_gt[i])
        # TP only if overlap clears the threshold AND that GT isn't already taken
        if gt >= 0 and best_iou[i] >= iou_threshold and not matched[gt]:
            TP[i] = 1
            matched[gt] = True # claim that GT so nothing else can
        else:
            FP[i] = 1 # too loose, or a duplicate

    cum_TP = torch.cumsum(TP, dim=0) # cumulative sum of TPs. cum_TP[i] = sum of TP for j<i+1
    cum_FP = torch.cumsum(FP, dim=0)  # cumulative sum of FPs. cum_FP[i] = sum of FP for j<i+1
    # TP, FP are in descending order of scores, so going start->end throw this list
    # gives us vectors of recalls and precisions tooken with different score thresholds
    # and here conf_score of each box(we iterate from 1->0) acts threshold value
    # confidence threshold 1 -> 0
    recall = cum_TP / (num_gt + 1e-9)  # TP / (TP + FN) how many GT objects model found : 0 -> 1 monotonic
    precision = cum_TP / (cum_TP + cum_FP + 1e-9)  # TP / (TP + FP) throw all found object how many are true : 1 -> 0 not monotonic
    return every_point_interpolation(recall, precision)


def every_point_interpolation(recall, precision):
    # puts start/end boundaries for integral
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    # for every recall we want max precision value where conf_thrs is no bigger then thrs for recall
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # tuple where each nonzero value is tenzor(i, j) index.
    # So it returns indexes where recall[i] != recall[i+1]
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0] + 1
    ap = torch.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap.item()  # final integral of RP-curve (taken in few points, not continuous)


def mean_average_precision(dets, gts, num_cls=20):
    iou_thresholds = torch.linspace(0.5, 0.95, 10)

    if gts.shape[0] == 0:
        return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_5095": 0.0}
    present_classes = gts[:, 5].unique().tolist()

    ap_table = []  # rows = classes with GT, cols = thresholds
    for c in present_classes:
        dets_c = dets[dets[:, 6] == c]
        gts_c = gts[gts[:, 5] == c]
        num_gt = gts_c.shape[0]

        best_iou, best_gt = class_match_ious(dets_c, gts_c)

        # find AP for this class at each IoU threshold
        aps = [ap_from_matches(best_iou, best_gt, num_gt, t.item())
               for t in iou_thresholds]
        ap_table.append(aps)

    if len(ap_table) == 0:
        return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_5095": 0.0}

    ap_table = torch.tensor(ap_table)

    return {
        "mAP_50":   ap_table[:, 0].mean().item(),   # mean over classes IoU with GT 0.50
        "mAP_75":   ap_table[:, 5].mean().item(),   # mean over classes IoU with GT IoU 0.75
        "mAP_5095": ap_table.mean().item(),         # mean over classes AND thresholds 0.50->0.95
    }