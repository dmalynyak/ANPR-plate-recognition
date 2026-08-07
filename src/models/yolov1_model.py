import time

import torch
from torch import nn

'''
target(7,7,25): for target[i][j]  [0:20] class one-hot, [20] objectness, [21:25] x,y,w,h
prediction(7,7,30): for target[i][j]  [0:20] class one-hot, [20] confidence, [21:25] x,y,w,h, [25] confidence, [26:30] xywh

-----------  loss function  --------------
i - loop for cells (1-49)
j - llop for boxes in cell (1-2)
λ_coord = 5
λ_noobj = 0.5

indicators functions: 
1_obj_ij := 1 if box j is responsible (highest IoU) for cell i (cell must contain object)
1_noobj_ij := 1 if box j is NOT responsible (lowest IoU) for cell i (cell contains object) and 1 for every box in every cell that does not contain object
1_obj_i := 1 if cell i contains an object

squared errors:
center error:= λ_coord * (Sum over every box in every cell)[1_obj_ij * ((x-x̂)² + (y−ŷ)²)]
size error:= λ_coord * (Sum over every box in every cell)[1_obj_ij * ((√w−√ŵ)² + (√h−√ĥ)²)]
confidence, object, responsible:= (Sum over every box in every cell)[1_obj_ij * (C − Ĉ)²]
confidence, unresponsible:=  λ_noobj*(Sum over every box in every cell)[1_noobj_ij * (C − Ĉ)²]
classification:= (Sum over every cell)[1_obj_i * (Sum over every classes)[p(c) − p̂(c)]²]

explanation: 
center error - pushes predicted center to ground truth center. Only for responsible box in cell that contains object
size error - pushes predicted width/heigh to ground truth. Only for responsible box in cell that contains object
confidence, object - for cell with object pushes Ĉ to truth (IoU of (x̂, ŷ, ŵ, ĥ) and (x, y, w, h)) for responsible box
confidence, unresponsible - for cell with object pushes Ĉ to truth (0) for loser box in object cell and for all boxes in noobject cells
classification - for cell with object pushes class prediction to truth (0,..,0,1,0,..,0). boxes share class prediction

loss function is sum of five squarred errors

prediction = (B, 7, 7, 2*5 + 20)
target = (B, 7, 7, 25) = (B, cellx, celly, x+y+w+h+o+20)
target(7,7,25): for target[i][j]  [0:20] class one-hot   [20] objectness   [21:25] x,y,w,h
prediction(7,7,30): for target[i][j]  [0:20] class one-hot, [20] confidence, [21:25] x,y,w,h, [25] confidence, [26:30] xywh

tensor criterion implementation
(B, 7, 7, 30) = prediction
   ->
predictions[:, :, :, 21:25] - box0 xywh
predictions[:, :, :, 26:30] - box1 xywh
predictions[:, :, :, 20:21] - box0 confidence
predictions[:, :, :, 25:26] - box1 confidence
predictions[:, :, :, 0:20]  - classes

(B, 7, 7, 1) = computed
    ->
iou0(B, 7, 7, 1) - IOU of box0 and target for every cell
iou1(B, 7, 7, 1) - IOU of box1 and target for every cell
best_box(B, 7, 7, 1) - responsible box (highes IOU of box and target for every cell)
best_iou(B, 7, 7, 1) - IOU of best_box and target for every cell

(B, 7, 7, x) = from predictions/iou/best
    ->
resp_xy(B, 7, 7, 2) - x,y of responsible box for every cell
resp_wh(B, 7, 7, 2) - w,h of responsible box for every cell
resp_conf(B, 7, 7, 1) - confidence of responsible box for every cell
'''

#----------------------  YOLOv1 forward pass  ------------------------

# block for convolution -> batch normalization -> relu
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),  # description is in crnn_algorithm
            nn.LeakyReLU(0.1),  # negative values becomes 0.1*x not 0
        )

    def forward(self, x):
        return self.conv(x)


# block for backbone (feature extraction) basic CNN
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # self.cnn = nn.Sequential(
        #     ConvBlock(3, 32, (3, 3), 1, 1),  # (B, 3, 448, 448) -> (B, 32, 448, 448)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 32, 448, 448) -> (B, 32, 224, 224)
        #     ConvBlock(32, 64, (3, 3), 1, 1),  # (B, 32, 224, 224) -> (B, 64, 224, 224)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 64, 224, 224) -> (B, 64, 112, 112)
        #     ConvBlock(64, 128, (3, 3), 1, 1),  # (B, 64, 112, 112) -> (B, 128, 112, 112)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 128, 112, 112) -> (B, 128, 56, 56)
        #     ConvBlock(128, 256, (3, 3), 1, 1),  # (B, 128, 56, 56) -> (B, 256, 56, 56)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 256, 56, 56) -> (B, 256, 28, 28)
        #     ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 28, 28) -> (B, 512, 28, 28)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 512, 28, 28) -> (B, 512, 14, 14)
        #     ConvBlock(512, 1024, (3, 3), 1, 1),  # (B, 512, 14, 14) -> (B, 1024, 14, 14)
        #     nn.MaxPool2d((2, 2), (2, 2)),  # (B, 1024, 14, 14) -> (B, 1024, 7, 7)
        # )
        self.cnn = nn.Sequential(
            ConvBlock(3, 64, (7, 7), 2, 3),  # (B, 3, 448, 448) -> (B, 64, 224, 224)
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 64, 224, 224) -> (B, 64, 112, 112)
            ConvBlock(64, 192, (3, 3), 1, 1),  # (B, 64, 112, 112) -> (B, 192, 112, 112)
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 192, 112, 112) -> (B, 192, 56, 56)

            ConvBlock(192, 128, (1, 1), 1, 0),  # (B, 192, 56, 56) -> (B, 128, 56, 56)
            ConvBlock(128, 256, (3, 3), 1, 1),  # (B, 128, 56, 56) -> (B, 256, 56, 56)
            ConvBlock(256, 256, (1, 1), 1, 0),  # (B, 256, 56, 56) -> (B, 256, 56, 56)
            ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 56, 56) -> (B, 512, 56, 56)
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 512, 56, 56) -> (B, 512, 28, 28)

            ConvBlock(512, 256, (1, 1), 1, 0),  # (B, 512, 28, 28) -> (B, 256, 28, 28)
            ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 28, 28) -> (B, 512, 28, 28)
            ConvBlock(512, 256, (1, 1), 1, 0),  # (B, 512, 28, 28) -> (B, 256, 28, 28)
            ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 28, 28) -> (B, 512, 28, 28)
            ConvBlock(512, 256, (1, 1), 1, 0),  # (B, 512, 28, 28) -> (B, 256, 28, 28)
            ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 28, 28) -> (B, 512, 28, 28)
            ConvBlock(512, 256, (1, 1), 1, 0),  # (B, 512, 28, 28) -> (B, 256, 28, 28)
            ConvBlock(256, 512, (3, 3), 1, 1),  # (B, 256, 28, 28) -> (B, 512, 28, 28)
            ConvBlock(512, 512, (1, 1), 1, 0),  # (B, 512, 28, 28) -> (B, 512, 28, 28)
            ConvBlock(512, 1024, (3, 3), 1, 1),  # (B, 512, 28, 28) -> (B, 1024, 28, 28)
            nn.MaxPool2d((2, 2), (2, 2)),  # (B, 1024, 28, 28) -> (B, 1024, 14, 14)

            ConvBlock(1024, 512, (1, 1), 1, 0),  # (B, 1024, 14, 14) -> (B, 512, 14, 14)
            ConvBlock(512, 1024, (3, 3), 1, 1),  # (B, 512, 14, 14) -> (B, 1024, 14, 14)
            ConvBlock(1024, 512, (1, 1), 1, 0),  # (B, 1024, 14, 14) -> (B, 512, 14, 14)
            ConvBlock(512, 1024, (3, 3), 1, 1),  # (B, 512, 14, 14) -> (B, 1024, 14, 14)
            ConvBlock(1024, 1024, (3, 3), 1, 1),  # (B, 1024, 14, 14) -> (B, 1024, 14, 14)
            ConvBlock(1024, 1024, (3, 3), 2, 1),  # (B, 1024, 14, 14) -> (B, 1024, 7, 7)

            ConvBlock(1024, 1024, (3, 3), 1, 1),  # (B, 1024, 7, 7) -> (B, 1024, 7, 7)
            ConvBlock(1024, 1024, (3, 3), 1, 1),  # (B, 1024, 7, 7) -> (B, 1024, 7, 7)

        )

    def forward(self, x):
        return self.cnn(x)


# block for flattening that MLP them reshaping
class FC(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.fc_1 = nn.Linear(in_channels, mid_channels)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.act(x) # activation
        x = self.drop(x)
        x = self.fc_2(x)
        x = x.view(-1, 7, 7, 30)
        return x


# algorithm for forward pass YOLOv1 (B, 3, 448, 448) -> (B, 7, 7, 30)
class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.fc = FC(50176, 4096,1470)

    def forward(self, x):
        x = self.backbone(x)  # self.backbone(x)   →   nn.Module.__call__   →   Backbone.forward(x)
        x = self.fc(x)
        return x  # (B, 7, 7, 30)

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