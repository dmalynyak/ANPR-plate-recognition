import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from src.parsing import load_cache_nomeroff, PlateDataset

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


# ------- training --------------


CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# encodes names-strings into integer indices
def encode_targets(labels):  # labels = ["AB12", "XYZ"]
    flat, lengths = [], []
    for s in labels:
        flat.extend(CHARS.index(c) + 1 for c in s)  # +1 because blank=0
        lengths.append(len(s))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # enables model to train (just changes flag)
    total = 0.0  # accumulates loss for every batch in epoch. interface/debug
    for images, labels in loader:
        images = images.to(device)  # puts images into GPU memmory, so model(images) are performed using GPU
        log_probs = model(images)  # forward pass
        T, B, _ = log_probs.shape  # (T, B, C)

        targets, target_lengths = encode_targets(labels)  # makes one_hot from plain text
        input_lengths = torch.full((B,), T, dtype=torch.long)  # tells torch timesteps of sample

        loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)

        optimizer.zero_grad()  # cleans gradient for correct gradient descent
        loss.backward()  # pytorch computes gradient using backpropagation
        optimizer.step()  # makes one step of gradient descent
        total += loss.item()  # converts tensor into float value

    return total / len(loader)  # convention


# ---------- decoding ---------- (parce final prediction from algorithm results)


# decods one image. from CTC results to actual label
def decode_image(log_probs):  # (32, B, 37)
    idx = log_probs.argmax(2)  # (32, B) - value is index
    out = []
    for b in range(idx.shape[1]):  # loop for batches
        seq, prev = [], -1  # -1 so prev != first
        for t in range(idx.shape[0]):  # loop for timesteps
            i = idx[t, b].item()  # extracts value
            if i != prev and i != 0:
                seq.append(CHARS[i - 1])
            prev = i
        out.append("".join(seq))
    return out


# decods only one batch, so it is much faster (interface/debug)
# prints only first name
def decode_batch(model, loader):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(loader))  # iter - iterator for loader, next - first batch
        preds = decode_image(model(images))
        print(f"prediction: {preds[0]}, actual: {labels[0]}")

    return preds


# # predict one image
# def predict_image(model, path, device, w=128, h=32):
#     img, true_name = parsing.img_parcing(path, w, h)  # (32,128)
#     t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
#     # (32,128) -> (1,32,128) -> (1,1,32,128) = (B,C,H,W) with B=1
#
#     model.eval()
#     with torch.no_grad():
#         pred = decode_image(model(t.to(device)))[0]  # first (and only) item in the batch
#
#     print(f"{path}  pred: {pred!r}  true: {true_name!r}")
#     return pred


# ---------- testing and validation -------------------


def validation(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:  # loop across batches
            images = images.to(device)
            log_probs = model(images)
            T, B, _ = log_probs.shape

            targets, target_lengths = encode_targets(labels)
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = criterion(log_probs.cpu(), targets, input_lengths,
                             target_lengths)  # log_probs.cpu() copies data from GPU to RAM

            total += loss.item()  # already normalized loss across images in batch (mean)

    return total / len(loader)  # sum of loss of all batches / number of batches


def test_model(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_symbols, correct_symbols = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            pred = decode_image(log_probs.cpu())

            for p, s in zip(pred, labels):
                total += 1
                if p == s:
                    correct += 1

                correct_symbols += sum(a == b for a, b in zip(p, s))
                total_symbols += len(s)

    return correct / total, correct_symbols / total_symbols


def test_model_debug(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_symbols, correct_symbols = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            pred = decode_image(log_probs)

            for p, s in zip(pred, labels):
                total += 1
                if p == s:
                    correct += 1
                else:
                    print(f"prediction: {p}, actual: {s}")

                correct_symbols += sum(a == b for a, b in zip(p, s))
                total_symbols += len(s)

    return correct / total, correct_symbols / total_symbols


def main():
    # activates Metal Performance Shaders (gpu) in apple m-proccessor
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # enable GPU acceleration (GPU computing)
    device = torch.device("cpu")
    print("using", device)

    # X_train, Y_train = parcing.load_dataset_kaggle("dataset/ocr_kaggle/train", width = 128, heigh = 32)
    # X_val, Y_val = parcing.load_dataset_kaggle("dataset/ocr_kaggle/val", width = 128, heigh = 32)
    # X_test, Y_test = parcing.load_dataset_kaggle("dataset/ocr_kaggle/test", width = 128, heigh = 32)

    X_train, Y_train = load_cache_nomeroff("dataset/ocr_nomeroff/train/img", "dataset/ocr_nomeroff/train/ann",
                                                   width=128, heigh=32, cache_name="cache/train")
    X_val, Y_val = load_cache_nomeroff("dataset/ocr_nomeroff/val/img", "dataset/ocr_nomeroff/val/ann",
                                               width=128, heigh=32, cache_name="cache/val")
    X_test, Y_test = load_cache_nomeroff("dataset/ocr_nomeroff/test/img", "dataset/ocr_nomeroff/test/ann",
                                                 width=128, heigh=32, cache_name="cache/test")

    train_loader = DataLoader(PlateDataset(X_train, Y_train, expansion=True), batch_size=32,
                              shuffle=True)  # loads bathes into train_epoch
    val_loader = DataLoader(PlateDataset(X_val, Y_val, expansion=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(PlateDataset(X_test, Y_test, expansion=False), batch_size=32, shuffle=False)

    model = CRNN().to(device)  # copies all weights and biases into GPU memmory
    criterion = nn.CTCLoss(blank=0,
                           zero_infinity=True)  # does not have GPU implementation, so this is only evaluation done by cpu
    # optimizer for gradient descent.
    # model.parameters() - all learnable weight/bias tensors, so the optimizer knows what to update
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0
    start = time.perf_counter()
    for epoch in range(30):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if epoch % 5 == 0:
            end = time.perf_counter()
            elapsed = end - start
            val_loss = validation(model, val_loader, criterion, device)
            p, c = test_model(model, val_loader, device)
            print(
                f"epoch {epoch}: train loss: {loss:.4f}  val loss: {val_loss:.4f}  plate acc:{p:.3f}  char acc:{c:.3f}, time: {elapsed:.2f}")
            start = time.perf_counter()

            if p > best_acc:  # when accuracy based on test dataset beats previous, model get downloaded
                best_acc = p
                torch.save(model.state_dict(), "../../models_test/foo.pt")
                print(f"  saved, (accuracy on validation dataset: {p:.3f})")

    model.load_state_dict(torch.load("../../models_test/foo.pt", map_location=device))
    total, correct = test_model_debug(model, test_loader, device)
    print(f"correct plates: {total}, correct symbols: {correct}")


if __name__ == "__main__":
    main()