import time

import torch
from torch import nn

# -----------------------  LOSS function and helper functions  ----------------


#           (B, 7, 7, 4)
def get_iou(box0, box1):
    b0x1, b0y1 = box0[:, :, :, 0:1] - box0[:, :, :, 2:3]/2, box0[:, :, :, 1:2] - box0[:, :, :, 3:4]/2
    b0x2, b0y2 = box0[:, :, :, 0:1] + box0[:, :, :, 2:3]/2, box0[:, :, :, 1:2] + box0[:, :, :, 3:4]/2
    b1x1, b1y1 = box1[:, :, :, 0:1] - box1[:, :, :, 2:3]/2, box1[:, :, :, 1:2] - box1[:, :, :, 3:4]/2
    b1x2, b1y2 = box1[:, :, :, 0:1] + box1[:, :, :, 2:3]/2, box1[:, :, :, 1:2] + box1[:, :, :, 3:4]/2

    intersection = (torch.min(b0x2, b1x2) - torch.max(b0x1, b1x1)).clamp(0) * (torch.min(b0y2, b1y2) - torch.max(b0y1, b1y1)).clamp(0)
    area = box0[:, :, :, 2:3]*box0[:, :, :, 3:4] + box1[:, :, :, 2:3]*box1[:, :, :, 3:4]

    return intersection / (area - intersection + 1e-6)

#            (B,7,7,4)(B,1,7,1)(B,7,1,1)
def cell_to_img(box, rows, cols): # transforms cell relative x,y from 'taget' to image relative for 'prediction'
#(B,7,7,1)=(B,1,7,1)+(B,7,7,1)
    x = (cols + box[:, :, :, 0:1]) / 7
    y = (rows + box[:, :, :, 1:2]) / 7
#   (B,7,7,4) =             (B,7,7,1)+(B,7,7,1)+(B,7,7,1)+(B,7,7,1)
    return torch.cat([x, y, box[:, :, :, 2:3], box[:, :, :, 3:4]], dim = -1)


def criterion(predictions, targets):
    l_coord = 5.0
    l_noobj = 0.5

    batch_size = predictions.size(0)
    device = predictions.device
    cols = torch.arange(7, device=device).view(1, 1, 7, 1) #  column index (x) [0][0][col][0]
    rows = torch.arange(7, device=device).view(1, 7, 1, 1) #  row index (y)

    iou0 = get_iou(cell_to_img(predictions[..., 21:25], rows, cols), cell_to_img(targets[:, :, :, 21:25], rows, cols))
    iou1 = get_iou(cell_to_img(predictions[..., 26:30], rows, cols), cell_to_img(targets[:, :, :, 21:25], rows, cols))
    best_iou = torch.max(iou0, iou1) # finds max value for everu cell

    # mask -def- tensor with booleans
    objectness_mask = targets[:, :, :, 20:21]
    best_box = (iou0 < iou1).float()  # 1.0 if iou0<iou1 and 0.0 otherwise

    resp_xy = (1 - best_box) * predictions[..., 21:23] + best_box * predictions[..., 26:28]
    resp_wh = ((1 - best_box) * predictions[..., 23:25] + best_box * predictions[..., 28:30])
    resp_conf = (1 - best_box) * predictions[..., 20:21] + best_box * predictions[..., 25:26]

    # 1 center error
    center_error = l_coord * (torch.square(objectness_mask * (resp_xy - targets[:, :, :, 21:23])).sum(axis=-1, keepdim=True)).sum()
    # 2 size error
    size_sqrt_pred = torch.sign(resp_wh) * torch.sqrt(resp_wh.abs() + 1e-6)
    size_sqrt_target= torch.sqrt(targets[:, :, :, 23:25] + 1e-6)
    size_error = l_coord * (torch.square(objectness_mask * (size_sqrt_pred - size_sqrt_target)).sum(axis=-1, keepdim=True)).sum()
    # 3 confidence object responsible error
    conf_obj_error = torch.square(objectness_mask * (resp_conf - best_iou.detach())).sum()
    # 4 confidence unresponsible error
    unresp0 = 1 - objectness_mask * (1 - best_box) # box0 is not responsible (or object does not exist)
    unresp1 = 1 - objectness_mask * best_box # box1 is not responsible here (or object does not exist)
    conf_noobj_error = l_noobj * (torch.square(unresp0 * predictions[..., 20:21]).sum() + torch.square(unresp1 * predictions[..., 25:26]).sum()
    )
    # 5 classification error
    class_error = torch.square(objectness_mask * (predictions[:, :, :, 0:20] - targets[:, :, :, 0:20])).sum(axis=-1, keepdim=True).sum()

    return (center_error + size_error + conf_obj_error + conf_noobj_error + class_error) / batch_size


# -------------------  train/test/val  --------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    start = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images) #  (B, 7, 7, 30)

        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if i % 20 == 0:
            torch.save(model.state_dict(), "models_test/yolov1_foo/latest.pt")
            end = time.perf_counter()
            print(f"saved latest: batch {i}/{len(loader)}  loss {loss.item():.3f} time: {end - start:.3f} \n")
            start = time.perf_counter()

    return epoch_loss / len(loader)

def test_model(model, loader, device, conf_threshold, iou_threshold):
    model.eval()
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images) # (B, 7, 7, 30)

            conf0 = predictions[..., 20:21]  # (B,7,7,1)
            conf1 = predictions[..., 25:26]  # (B,7,7,1)
            box0 = predictions[..., 21:25]   # (B,7,7,4)
            box1 = predictions[..., 26:30]   # (B,7,7,4)

            best_conf = torch.max(conf0, conf1)
            best_box = torch.where(conf0 > conf1, box0, box1) # ((B,7,7,4))

            cols = torch.arange(7, device=device).view(1, 1, 7, 1)  # column index (x) [0][0][col][0]
            rows = torch.arange(7, device=device).view(1, 7, 1, 1)  # row index (y)
            best_iou = get_iou(cell_to_img(best_box, rows, cols),  cell_to_img(targets[:, :, :, 21:25], rows, cols))

            # masks
            obj = targets[..., 20:21] > 0.5  # objectness mask
            mod_positive = best_conf > conf_threshold  # all positive objects model predicts (they could be false positive)
            true_positive = best_iou > iou_threshold  # all well-localized objects model predicts (model can be unconfident)

            TP += (true_positive & mod_positive).sum() # number of true positive answers model predicts and confident (correct)
            FP += (mod_positive & ~true_positive).sum() # number of false positive answers model predicts and confident (incorrect)
            FN += (~mod_positive & obj).sum() # number of exists objects model not confident (incorrect)
            TN += (~mod_positive & ~obj).sum() # number of not exists objects model not confident (correct)

        accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        F1 = 2 * precision * recall / (precision + recall + 1e-6)

        return accuracy.item(), precision.item(), recall.item(), F1.item()

def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)

            loss = criterion(predictions, targets) / predictions.size(0)
            total += loss

    return total / len(loader)