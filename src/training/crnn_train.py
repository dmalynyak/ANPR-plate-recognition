import time, torch, argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from src.train_architecture import crnn_train_architecture
from src.models import crnn_model
from src.data_loaders import crnn_dataload
from src import parsing

def resolve_device(name):
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device '{name}'")



def train_logic(save_path, device):
    print("using", device)

    # training with kaggle dataset (588 images with augmentation)
    # X_train, Y_train = crnn_dataload.load_dataset_kaggle("data/dataset/ocr_kaggle/train", width = 128, heigh = 32)
    # X_val, Y_val = crnn_dataload.load_dataset_kaggle("data/dataset/ocr_kaggle/val", width = 128, heigh = 32)
    # X_test, Y_test = crnn_dataload.load_dataset_kaggle("data/dataset/ocr_kaggle/test", width = 128, heigh = 32)

    # training with nomeroff dataset (40,000 images)
    X_train, Y_train = parsing.load_cache_nomeroff("data/dataset/ocr_nomeroff/train/img", "dataset/ocr_nomeroff/train/ann", width = 128, heigh = 32, cache_name = "cache/train")
    X_val, Y_val = parsing.load_cache_nomeroff("data/dataset/ocr_nomeroff/val/img", "dataset/ocr_nomeroff/val/ann", width = 128, heigh = 32, cache_name = "cache/val")
    X_test, Y_test = parsing.load_cache_nomeroff("data/dataset/ocr_nomeroff/test/img", "dataset/ocr_nomeroff/test/ann", width = 128, heigh = 32, cache_name = "cache/test")


    train_loader = DataLoader(crnn_dataload.CRNNDataset(X_train, Y_train, expansion=True), batch_size=16, shuffle=True) # loads bathes into train_epoch
    val_loader = DataLoader(crnn_dataload.CRNNDataset(X_val, Y_val, expansion=False), batch_size=16, shuffle=False)
    test_loader = DataLoader(crnn_dataload.CRNNDataset(X_test, Y_test, expansion=False), batch_size=16, shuffle=False)


    model = crnn_model.CRNN().to(device) # copies all weights and biases into GPU memmory
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # does not have GPU implementation, so this is only evaluation done by cpu
    # optimizer for gradient descent.
    # model.parameters() - all learnable weight/bias tensors, so the optimizer knows what to update
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    best_acc = 0.0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(50):
        # actual training
        loss = crnn_train_architecture.train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # information about training process
        if epoch % 5 == 0:
            end = time.perf_counter()
            elapsed = end - start
            val_loss = crnn_train_architecture.validation(model, val_loader, criterion, device)
            p, c = crnn_train_architecture.test_model(model, val_loader, device)
            print(f"epoch {epoch}: train loss: {loss:.4f}  val loss: {val_loss:.4f}  plate acc:{p:.3f}  char acc:{c:.3f}, time: {elapsed:.2f}")
            start = time.perf_counter()

            if p > best_acc: # when accuracy based on test dataset beats previous, model get downloaded
                best_acc = p
                torch.save(model.state_dict(), f"{save_path}/best.pt")
                print(f" saved {save_path}_best.pt, (accuracy on validation dataset: {p:.3f})")

    model.load_state_dict(torch.load(f"{save_path}/best.pt", map_location=device))
    total, correct = crnn_train_architecture.test_model_debug(model, test_loader, device)
    print(f"correct plates: {total}, correct symbols: {correct}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-path", help="path to folder where model states will be saved")
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_logic(args.save_path, device)


if __name__ == "__main__":
    main()