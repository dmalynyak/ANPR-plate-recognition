from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), # Batch normalization sustracts all biases so they are redundant here
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.conv(x)


# backbone for both classification (pre-training) and detection
class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()

        # comment are structure for classification (pre-training) backbone ImageNet 224x224
        self.part1 = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # 224 -> 224
            nn.MaxPool2d(kernel_size=2, stride=2), # 224 -> 112

            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 112 -> 112
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 56 -> 56
            ConvBlock(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),  # 56 -> 56
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 56 -> 56
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 28 -> 28
            ConvBlock(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),  # 28 -> 28
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 28 -> 28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # 14 -> 14
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # 14 -> 14
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            # this part is the part we passthrow to DetectHead for concatination
        )

        self.part2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7 or (B, 516, 26, 26) -> (B, 516, 13, 13)

            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # 7 -> 7
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),  # 7 -> 7
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 7 -> 7
        )

    def forward(self, x):
        passthrow = self.part1(x)
        out = self.part2(passthrow)
        return out, passthrow


# input: (B, 1024, 7, 7)
class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0), # 1024,7 -> 1000,7 without Batch normalization and MaxPooling
            nn.AdaptiveAvgPool2d(1), # input-whatever, output (1, 1) 1000,7,7 -> 1000,1,1
            nn.Flatten(), # B,1000,1,1 -> B,1000
            # nn.Softmax(dim=1), # CCE applies log-softmax
        )
    def forward(self, x):
        return self.head(x)


# input: (B, 1024, 13, 13)
class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.rearrange = nn.PixelUnshuffle(2) # (512, 26, 26) -> (2048, 13, 13)
        self.conv3 = ConvBlock(in_channels=1024 + 2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=125, kernel_size=1, stride=1, padding=0)

    def forward(self, x, passthrow_raw):
        x = self.conv1(x)   #(B, 1024, 13, 13)
        x = self.conv2(x)   #(B, 1024, 13, 13)
        passthrow = self.rearrange(passthrow_raw)    #(B, 512, 26, 26) -> (B, 2048, 13, 13)
        x = torch.cat([x, passthrow], dim=1)    #(B, 1024+2048, 13, 13)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class YOLOv2Pretraining(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Darknet19()
        self.head = ClassificationHead()

    def forward(self, x):
        x, _ = self.backbone(x)
        x = self.head(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Darknet19()
        self.head = DetectionHead()

    def forward(self, x):
        x, passthrow = self.backbone(x)
        x = self.head(x, passthrow)
        return x