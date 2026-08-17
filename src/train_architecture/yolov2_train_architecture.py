import os
import time, torch
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm


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
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.replace('\x00', '').strip()
                if not line.split():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    w = float(parts[-2])
                    h = float(parts[-1])
                    boxes.append([w, h, -1])
                except ValueError:
                    continue

    anchors = 13 * torch.rand(5, 2)

    anch_iter = 0
    for passes in range(100):
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
                gcx = gt[..., 0] + cx
                gcy = gt[..., 1] + cy
                b1x1, b1y1 = gcx - gt[..., 2] / 2, gcy - gt[..., 3] / 2
                b1x2, b1y2 = gcx + gt[..., 2] / 2, gcy + gt[..., 3] / 2

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

def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=3, min_lr_ratio=0.01):
    base_lr      = optimizer.param_groups[0]["lr"]
    total_steps  = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)

    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                               eta_min=base_lr * min_lr_ratio)

    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

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

        # if i % 500 == 0:
        #     torch.save(model.state_dict(), f"{save_path}/latest.pt")
        #     end = time.perf_counter()
        #     print(f"saved latest: batch {i}/{len(loader)}  loss {loss.item():.3f} time: {end - start:.3f} \n")
        #     start = time.perf_counter()

    return epoch_loss / len(loader)