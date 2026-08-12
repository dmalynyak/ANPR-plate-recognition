import os
import time, torch
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from src.data_loaders.yolov2_dataload import garbage_names_clean, load_one_image


'''
trining goals (targets to push prediction to during training):
anchors: ((pw0, ph0), (pw1, ph1),(pw2, ph2),(pw3, ph3),(pw4, ph4))
prediction: (13, 13, 5, [tx, ty, tw, th, to, so, s1, ...])
GT:         (13, 13, 5, [gx, gy, gw, gh, obj, g0, g1, ...])
𝜎(tx) -> gx
𝜎(ty) -> gy
tw -> ln (gw / pwk)
th -> ln (gh / phk)
𝜎(t0) -> IoU(GT, pred) - uncentered

loss:
every error is takes on every anchor in every cell
responsible:
    λcoord * (𝜎(tx) - gx)²
    λcoord * (𝜎(ty) - gy)²
    λcoord * (tw - ln(gw/pwk))²
    λcoord * (th - ln(gh/phk))²
    λobj   * (𝜎(to) - IoU(GT, pred))²
    λclass * ∑ᵢ(softmax(s)ᵢ - gᵢ)²
irresponsible:
    λnoobj * (𝜎(to) - 0)² - when IoU(GT, pred) < threshold
'''

# same units (cell used, so anchor should be devided by 13)
def get_iou_centered(box1, box2):
    intersection = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
    area = box1[0]*box1[1] + box2[0]*box2[1]

    return intersection / ( area  - intersection + 1e-6)

# same units
def get_iou_uncentered(box0, box1):
    b0x1, b0y1 = box0[0:1] - box0[2:3]/2, box0[1:2] - box0[3:4]/2
    b0x2, b0y2 = box0[0:1] + box0[2:3]/2, box0[1:2] + box0[3:4]/2
    b1x1, b1y1 = box1[0:1] - box1[2:3]/2, box1[1:2] - box1[3:4]/2
    b1x2, b1y2 = box1[0:1] + box1[2:3]/2, box1[1:2] + box1[3:4]/2

    intersection = (torch.min(b0x2, b1x2) - torch.max(b0x1, b1x1)).clamp(0) * (torch.min(b0y2, b1y2) - torch.max(b0y1, b1y1)).clamp(0)
    area = box0[2:3]*box0[3:4] + box1[2:3]*box1[3:4]

    return intersection / (area - intersection + 1e-6)

def get_anchors(labels_path):

    folder_path = Path(labels_path)
    boxes = []
    files = list(folder_path.iterdir())

    for file_path in tqdm(files, desc="reading labels for anchors"):
        with open(file_path, 'r') as file:
            for line in file:
                if not line.split():
                    continue
                parts = line.split()
                boxes.append([float(parts[-2]), float(parts[-1]), -1])

    anchors = 13 * torch.rand(5, 2)

    anch_iter = 0
    for passes in range(10):
        flag = False
        anch_change = 0
        anch_iter += 1
        for box in tqdm(boxes, desc=f"pass {anch_iter}", leave=False):
            iou = -1
            for idx, anchor in enumerate(anchors):
                new_iou = get_iou_centered(torch.tensor(box), anchor / 13)
                if new_iou > iou:
                    if box[2] != idx:
                        anch_change += 1
                        flag = True
                    box[2] = idx
                    iou = new_iou
        print(f"pass {anch_iter}: {anch_change} changed")
        if not flag:
            break

        anchors_new = torch.zeros(5, 3)
        for box in boxes:
            anchor_idx = box[2]
            anchors_new[anchor_idx][0] += box[0]
            anchors_new[anchor_idx][1] += box[1]
            anchors_new[anchor_idx][2] += 1 # number of boxes where anchor is responsible
        for idx in range(5):
            if anchors_new[idx, 2] == 0:
                continue
            for j in range(2):
                anchors[idx, j] = 13 * anchors_new[idx, j] / anchors_new[idx, 2] # makes cell-units for anchor/13 to work properly

    return anchors


class YOLOv2Loss(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors) # moves anchors to device and in state_dict
        self.l_coord = 1
        self.l_obj = 5
        self.l_noobj = 1
        self.l_class = 1


    # nn.Module calls __call__
    def forward(self, predictions, gt):
        def get_iou_live():
            with torch.no_grad():

                dec = torch.clone(predictions)  # makes independant copy
                cx = torch.arange(13, device=predictions.device).view(1, 1, 13, 1)  # (1, 1, 13, 1) broadcasts to (B, 13, 13, 5) where (B, 13, i, 5) = i
                cy = torch.arange(13, device=predictions.device).view(1, 13, 1, 1)  # (1, 13, 1, 1)
                pw = self.anchors[:, 0].view(1, 1, 1, 5)  # broadcasts to (B, 13, 13, 5)
                ph = self.anchors[:, 1].view(1, 1, 1, 5)

                dec[..., 0] = torch.sigmoid(predictions[..., 0]) + cx  # bx = σ(tx) + col index
                dec[..., 1] = torch.sigmoid(predictions[..., 1]) + cy
                dec[..., 2] = pw * torch.exp(predictions[..., 2])  # bw
                dec[..., 3] = ph * torch.exp(predictions[..., 3])  # bh

                b0x1, b0y1 = dec[..., 0] - dec[..., 2] / 2, dec[..., 1] - dec[..., 3] / 2
                b0x2, b0y2 = dec[..., 0] + dec[..., 2] / 2, dec[..., 1] + dec[..., 3] / 2
                b1x1, b1y1 = gt[..., 0] - gt[..., 2] / 2, gt[..., 1] - gt[..., 3] / 2
                b1x2, b1y2 = gt[..., 0] + gt[..., 2] / 2, gt[..., 1] + gt[..., 3] / 2

                intersection = (torch.min(b0x2, b1x2) - torch.max(b0x1, b1x1)).clamp(0) * (
                            torch.min(b0y2, b1y2) - torch.max(b0y1, b1y1)).clamp(0)
                area = dec[..., 2] * dec[..., 3] + gt[..., 2] * gt[..., 3]

                return intersection / (area - intersection + 1e-6)

        batch_size = predictions.size(0)

        objectness = gt[..., 4] # (B, 13, 13) objectness mask with 1/0 float values

        # helper tensors:
        pw = self.anchors[:, 0].view(1, 1, 1, 5) # broadcasts to (B, 13, 13, 5)
        ph = self.anchors[:, 1].view(1, 1, 1, 5) # broadcasts to (B, 13, 13, 5)
        iou = get_iou_live()

        center_error = self.l_coord * ( objectness.unsqueeze(-1) * (torch.sigmoid(predictions[..., 0:2]) - gt[..., 0:2]) ** 2).sum()
        offset_error_w = self.l_coord * ( objectness * (predictions[..., 2] - torch.log( gt[..., 2] / pw + 1e-6)) ** 2).sum()
        offset_error_h = self.l_coord * ( objectness * (predictions[..., 3] - torch.log( gt[..., 3] / ph + 1e-6)) ** 2).sum()
        confidence_obj_error = self.l_obj * ( objectness * (torch.sigmoid(predictions[..., 4]) - iou.detach()) ** 2).sum()
        class_error = self.l_class * (objectness.unsqueeze(-1) * (torch.softmax(predictions[..., 5:25], dim = -1) - gt[..., 5:25]) ** 2).sum()
        confidence_noobj_error = self.l_noobj * ( (1 - objectness) * (torch.sigmoid(predictions[..., 4])) ** 2).sum()

        return (center_error + offset_error_h + offset_error_w + confidence_obj_error + class_error + confidence_noobj_error) / batch_size

def get_warmup_schedualer(optimizer):
    warmup_lr_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3, # start at 0.1% of optimizer lr
        end_factor=1.0, # increases up to the full optimizer lr
        total_iters=1000, # over the first 1000 iterations
    )
    return warmup_lr_scheduler

def train_epoch(model, loader, criterion, optimizer, scheduler, save_path, device, epoch):
    model.train()
    epoch_loss = 0.0
    start = time.perf_counter()


    # mixed precision (AMP) = autocast + GradScaler
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # warmup learning rate. increases lr from 0.1%optim.lr up to 100&optim.lr throw 1000 steps
    warmup_lr_scheduler = scheduler # get_warmup_schedualer(optimizer) is used in yolov2_train.py

    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            predictions = model(images)  # runs in mixed precision(float32 default and float16 if operation allows worse precision)
            loss = criterion(predictions, targets)

        # scaler tries to make autocast training (mixed fp16/32) as precise as plain fp32
        # scales loss with some factor and with chain rule every param.grad gets scaled by the same factor
        scaler.scale(loss).backward()  # scales up values so operations in float16 does not underflow while backproping

        scaler.unscale_(optimizer) # unscales (in-place) param.grad for next step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # scale down param.grad if norm of gradient exceeds

        scaler.step(optimizer)  # scales grads values down with the same scale factor used in backprop, skips steps if inf/nan appeared
        scaler.update()  # adjust scale factor for next iter based on how much if/nan appears
        warmup_lr_scheduler.step() # increases lr during warmup

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

        if i % 500 == 0:
            torch.save(model.state_dict(), f"{save_path}/latest.pt")
            end = time.perf_counter()
            print(f"saved latest: batch {i}/{len(loader)}  loss {loss.item():.3f} time: {end - start:.3f} \n")
            start = time.perf_counter()

    return epoch_loss / len(loader)

def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                predictions = model(images)
                loss = criterion(predictions, targets)

            total += loss.item()

    return total / len(loader)


# saves current state of training. Evaluate after each epoch.
def save_model_state(model, optimizer, epoch, scheduler, anchors, save_path, name):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "anchors": anchors,
    }, f"{save_path}/{name}.pt")
    print(f"saved {name} model: epoch: {epoch}")


# loads all model training data. Used for resuming training
def load_model_state(model, optimizer, scheduler, load_path, device):
    checkpoint = torch.load(f"{load_path}.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    anchors = checkpoint["anchors"]
    return model, optimizer, scheduler, anchors, epoch

def save_model_weights(model, epoch, save_path, name):
    torch.save(model.state_dict(), f"{save_path}/{name}_epoch_{epoch}.pt")
    print(f"saved {save_path}/{name}/epoch_{epoch}.pt weights")

# gives final inference answer (before postprocessing)
# pred: (13, 13, 5, 25) decoded: (13, 13, 5, 25)
def decode_prediction(pred, anchors, device):
    with torch.no_grad():

        dec = torch.clone(pred) # makes independant copy
        S = 13

        cx = torch.arange(S).view(1, S, 1).to(device)  # (1, 13, 1) broadcasts to (13, 13, 5) where (13, i, 5) = i
        cy = torch.arange(S).view(S, 1, 1).to(device)  # (13, 1, 1)
        pw = anchors[:, 0].view(1, 1, 5)  # broadcasts to (13, 13, 5)
        ph = anchors[:, 1].view(1, 1, 5)

        dec[..., 0] = torch.sigmoid(pred[..., 0]) + cx  # bx = σ(tx) + col index
        dec[..., 1] = torch.sigmoid(pred[..., 1]) + cy
        dec[..., 2] = pw * torch.exp(pred[..., 2]) # bw
        dec[..., 3] = ph * torch.exp(pred[..., 3]) # bh
        dec[..., 4] = torch.sigmoid(pred[..., 4]) # confidence
        dec[..., 5:25] = torch.softmax(pred[..., 5:25], dim = -1)

        return dec

def iou_thrd_mask(box, boxes): # (4,) and (N, 4) returns (N,)
    x1 = torch.max(box[0], boxes[:, 0])  # (M,)
    y1 = torch.max(box[1], boxes[:, 1])  # (M,)
    x2 = torch.min(box[2], boxes[:, 2])  # (M,)
    y2 = torch.min(box[3], boxes[:, 3])  # (M,)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # (M,)

    area_box = (box[2] - box[0]) * (box[3] - box[1])  # scalar
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (M,)

    union = area_box + area_boxes - inter
    return inter / (union + 1e-9)

def nms(detections, iou_boxes_threshold=0.5):


    keep_det = []
    for cls in detections[:, 5].unique(): # loop (0-20) existing object classes

        cls_det = detections[detections[:, 5] == cls] # choode one class detections
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


def get_detected_boxes(pred, anchors, conf_threshold = 0.001, iou_boxes_threshold = 0.5):

    device = pred.device
    dec = decode_prediction(pred, anchors, device)

    obj = dec[..., 4] # (13, 13, 5)
    cls_probs = dec[..., 5:25] # (13, 13, 5, 20)
    cls_score, cls_id = cls_probs.max(dim=-1) # both (13, 13, 5)
    score = obj * cls_score # (13, 13, 5)

    x1 = dec[..., 0] - dec[..., 2] / 2
    y1 = dec[..., 1] - dec[..., 3] / 2
    x2 = dec[..., 0] + dec[..., 2] / 2
    y2 =dec[..., 1] + dec[..., 3] / 2

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

    return detected_boxes # (N, 6)

def get_target_boxes(label_path):

    device = "cpu"
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
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
    return boxes_tensor # (M, 5)

def get_all_boxes(model, anchors, img_dir_path, label_dir_path, device, conf_threshold = 0.001, iou_boxes_threshold = 0.5):
    all_names = sorted(os.listdir(img_dir_path))
    good_names = garbage_names_clean(all_names, label_dir_path)

    detected_boxes = [] # (Total, N, 7) [img_id, x1[0], y1[0], x2[0], y2[0], score[0], class_id[0]]
    target_boxes = []   # (Total, M, 5) [img_id, x1, y1, x2, y2]
    for img_id, name in enumerate(good_names):
        img_path = os.path.join(img_dir_path, name)
        label_path = os.path.join(label_dir_path, os.path.splitext(name)[0] + '.txt')
        img = load_one_image(img_path, 416, device)

        with torch.no_grad():
            pred = model(img)[0].to("cpu")  # (13, 13, 5, 25)

        one_detected_boxes = get_detected_boxes(pred, anchors, conf_threshold, iou_boxes_threshold) # (N, 6)
        idx = torch.full((one_detected_boxes.shape[0], 1), float(img_id), dtype=one_detected_boxes.dtype, device=one_detected_boxes.device)
        one_detected_boxes = torch.cat([idx, one_detected_boxes], dim=1)
        detected_boxes.append(one_detected_boxes)

        one_target_boxes = get_target_boxes(label_path) # (M, 5)
        idx = torch.full((one_target_boxes.shape[0], 1), float(img_id), dtype=one_detected_boxes.dtype, device=one_detected_boxes.device)
        one_target_boxes = torch.cat([idx, one_target_boxes], dim=1)
        target_boxes.append(one_target_boxes)

    return detected_boxes, target_boxes

def match_detections(detected_boxes, target_boxes, class_id, iou_threshold):

    dets = torch.cat(detected_boxes, dim=0)   # (Img_num * N, 7)
    gts = torch.cat(target_boxes,  dim=0)    # (Img_num * M, 6)

    dets_c = dets[dets[:, 6] == class_id]     # TP+FP detections one class in all images (K, 7)
    gts_c = gts[gts[:, 5] == class_id]       # GT one class in all images (T, 6)
    num_gt = gts_c.shape[0]                   # TP+FN total number of TRUE objects of one class

    # sort detection with score rating
    order = dets_c[:, 5].argsort(descending=True)
    dets_c = dets_c[order]

    # --- state we carry through the loop ---
    matched = torch.zeros(num_gt, dtype=torch.bool)   # how many GT boxes are claimed (TP)
    TP = torch.zeros(dets_c.shape[0])                 # TP detections
    FP = torch.zeros(dets_c.shape[0])                 # FP detections


    for i in range(dets_c.shape[0]):
        det = dets_c[i] # one row (one detected box)
        img_id = det[0]

        gt_mask = gts_c[:, 0] == img_id # mask of real box in image det
        gt_idx  = gt_mask.nonzero(as_tuple=True)[0]   # rows in gts_c / matched
        gt_img  = gts_c[gt_mask]

        if gt_img.shape[0] == 0: # no GT here -> false positive
            FP[i] = 1
            continue

        # IoU of this detection against each GT box in the image (vectorized)
        ious = iou_thrd_mask(det[1:5], gt_img[:, 1:5])
        best_iou, best = ious.max(dim=0)              # best-overlapping GT + its index

        # TP only if overlap clears the threshold AND that GT isn't already taken
        if best_iou.item() >= iou_threshold and not matched[gt_idx[best]].item():
            TP[i] = 1
            matched[gt_idx[best]] = True              # claim that GT so nothing else can
        else:
            FP[i] = 1                                 # too loose, or a duplicate

    return TP, FP, num_gt # TP/FP detection mask of detected boxes and num_gt number of total GT boxes (for one class in all images at once)

def every_point_interpolation(recall, precision):
    # puts start/end boundaries for integral
    mrec = torch.cat([torch.tensor([0.0]), recall,    torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    # for every recall we want max precision value where conf_thrs is no bigger then thrs for recall
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # tuple where each nonzero value is tenzor(i, j) index.
    # So it returns indexes where recall[i] != recall[i+1]
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0] + 1
    ap  = torch.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap.item() # final integral of RP-curve (taken in few points, not continuous)

def average_precision(detected_boxes, target_boxes, class_id, iou_threshold):
    # chains the three steps you built: match -> precision/recall -> area
    TP, FP, num_gt = match_detections(detected_boxes, target_boxes, class_id, iou_threshold)

    if num_gt == 0:  # no object of this class in GT in all images
        return None
    if TP.numel() == 0: # objects of this class exist but model did not detect any -> AP = 0
        return 0.0

    cum_TP = torch.cumsum(TP, dim=0)  # cumulative sum of TPs. cum_TP[i] = sum of TP for j<i+1
    cum_FP = torch.cumsum(FP, dim=0)  # cumulative sum of FPs. cum_FP[i] = sum of FP for j<i+1
    # TP, FP are in descending order of scores, so going start->end throw this list
    # gives us vectors of recalls and precisions tooken with different score thresholds
    # and here conf_score of each box(we iterate from 1->0) acts threshold value
    # confidence threshold 1 -> 0
    recall = cum_TP / (num_gt + 1e-9)  # TP / (TP + FN) how many GT objects model found : 0 -> 1 monotonic
    precision = cum_TP / (cum_TP + cum_FP + 1e-9) # TP / (TP + FP) throw all found object how many are true : 1 -> 0 not monotonic

    ap = every_point_interpolation(recall, precision)
    return ap


def mean_average_precision(detected_boxes, target_boxes, num_cls=20):
    iou_thresholds = torch.linspace(0.5, 0.95, 10)   # 0.50, 0.55, ..., 0.95

    ap_table = [] # rows = classes with GT, cols = thresholds
    for c in range(num_cls):
        # find AP for this class at each IoU threshold
        aps = [average_precision(detected_boxes, target_boxes, c, thrsh.item() ) for thrsh in iou_thresholds]
        if aps[0] is None: # no GT for this class
            continue
        ap_table.append(aps)

    if len(ap_table) == 0:
        return {"mAP_50": 0.0, "mAP_75": 0.0, "mAP_5095": 0.0}

    ap_table = torch.tensor(ap_table)    # (num_valid_classes, 10)

    return {
        "mAP_50":   ap_table[:, 0].mean().item(),   # mean over classes IoU with GT 0.50
        "mAP_75":   ap_table[:, 5].mean().item(),   # mean over classes IoU with GT IoU 0.75
        "mAP_5095": ap_table.mean().item(),         # mean over classes AND thresholds 0.50->0.95
    }

def val_map(model,anchors, img_dir_path, label_dir_path, device="cpu", conf_threshold=0.001, iou_boxes_threshold=0.5):
    detected_boxes, target_boxes = get_all_boxes(model, anchors, img_dir_path, label_dir_path, device, conf_threshold, iou_boxes_threshold)

    mAP = mean_average_precision(detected_boxes, target_boxes)

    print("mAP@0.5: {:.4f}  mAP@0.75: {:.4f}  mAP@[.5:.95]: {:.4f}".format(
        mAP["mAP_50"], mAP["mAP_75"], mAP["mAP_5095"]))
    return mAP
