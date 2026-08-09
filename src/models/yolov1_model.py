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
