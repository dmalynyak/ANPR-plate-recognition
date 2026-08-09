import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from src.parsing import load_cache_nomeroff

'''
B - batch
C/D - channel/depth (the same)
T/W - timestep/width/slices (the same)
H - height
C - classes/results neurons

input X        (B, 1, 32, 128)     B = batch size
CNN            (B, 512, 1, 32)     height crushed to 1, width crushes to 32
squeeze H      (B, 512, 32)
permute        (B, 32, 512)        32 slices/timesteps (width), 256 features (depth)
BiLSTM         (B, 32, 512)        512 = 2 × 256 (bidirection)
Linear         (B, 32, 37)         512 input neurons, 37 = 36 chars + blank results. Fully connected (LS without Softmax)
log_softmax    (B, 32, 37)
CTCLoss        scalar
'''

'''
criterion - CTC loss

'''


# ---------- CRNN -------------


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # (B, 1, 32, 128) -> (B, 64, 32, 128)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # kernel-(2, 2) (B, 64, 32, 128) -> (B, 64, 16, 64)
            nn.Conv2d(64, 128, 3, padding=1),  # (B, 64, 16, 64) -> (B, 128, 16, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 128, 16, 64) -> (B, 128, 8, 32)
            nn.Conv2d(128, 256, 3, padding=1),  # (B, 128, 8, 32) -> (B, 256, 8, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 256, 8, 32) -> (B, 256, 4, 32)
            nn.Conv2d(256, 512, 3, padding=1),  # (B, 256, 4, 32) -> (B, 512, 4, 32)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 1), (4, 1)),  # (B, 512, 4, 32) -> (B, 512, 1, 32)
        )

        # BiLST - rnn + bidirectional LSTM
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)  # (B, 32, 512) -> 2x(B, 32, 512) -> (B, 32, 512)

        # fc - fully connected (Linear Softmax but without Softmax!)
        self.fc = nn.Linear(512, 37)  # (B, 32, 512) -> (B, 32, 37)

        self.dropout = nn.Dropout(0.3)

    #  valuated during model(images)
    def forward(self, X):  # (B, 1, 32, 128)
        X = self.cnn(X)  # (B, 512, 1, 32)
        X = X.squeeze(2)  # (B, 512, 32)
        X = X.permute(0, 2, 1)  # (B, 32, 512)
        X, _ = self.rnn(X)  # (B, 32, 512)
        X = self.dropout(X)
        X = self.fc(X)  # (B, 32, 37) = (B, T, C) = (batch, timestep/slices, classes/results neurons)
        X = X.permute(1, 0, 2)  # (32, B, 37) = (T, B, C)
        return X.log_softmax(2)  # (T, B, C)