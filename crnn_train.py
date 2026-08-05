import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

import parcing
import crnn_algorithm


def main():

    # activates Metal Performance Shaders (gpu) in apple m-proccessor
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # enable GPU acceleration (GPU computing)
    device = torch.device("cpu")
    print("using", device)

    X_train, Y_train = parcing.load_dataset_kaggle("dataset/ocr_kaggle/train", width = 128, heigh = 32)
    X_val, Y_val = parcing.load_dataset_kaggle("dataset/ocr_kaggle/val", width = 128, heigh = 32)
    X_test, Y_test = parcing.load_dataset_kaggle("dataset/ocr_kaggle/test", width = 128, heigh = 32)

    # X_train, Y_train = parcing.load_cache_nomeroff("dataset/ocr_nomeroff/train/img", "dataset/ocr_nomeroff/train/ann", width = 128, heigh = 32, cache_name = "cache/train")
    # X_val, Y_val = parcing.load_cache_nomeroff("dataset/ocr_nomeroff/val/img", "dataset/ocr_nomeroff/val/ann", width = 128, heigh = 32, cache_name = "cache/val")
    # X_test, Y_test = parcing.load_cache_nomeroff("dataset/ocr_nomeroff/test/img", "dataset/ocr_nomeroff/test/ann", width = 128, heigh = 32, cache_name = "cache/test")



    train_loader = DataLoader(parcing.PlateDataset(X_train, Y_train, expansion=True),  batch_size=32, shuffle=True) # loads bathes into train_epoch
    val_loader = DataLoader(parcing.PlateDataset(X_val, Y_val, expansion=False),  batch_size=32, shuffle=False)
    test_loader = DataLoader(parcing.PlateDataset(X_test, Y_test, expansion=False), batch_size=32, shuffle=False)



    model = crnn_algorithm.CRNN().to(device) # copies all weights and biases into GPU memmory
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # does not have GPU implementation, so this is only evaluation done by cpu
    # optimizer for gradient descent.
    # model.parameters() - all learnable weight/bias tensors, so the optimizer knows what to update
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    best_acc = 0.0
    start = time.perf_counter()
    for epoch in range(30):
        loss = crnn_algorithm.train_epoch(model, train_loader, criterion, optimizer, device)

        if epoch % 5 == 0:
            end = time.perf_counter()
            elapsed = end - start
            val_loss = crnn_algorithm.validation(model, val_loader, criterion, device)
            p, c = crnn_algorithm.test_model(model, val_loader, device)
            print(f"epoch {epoch}: train loss: {loss:.4f}  val loss: {val_loss:.4f}  plate acc:{p:.3f}  char acc:{c:.3f}, time: {elapsed:.2f}")
            start = time.perf_counter()

            if p > best_acc: # when accuracy based on test dataset beats previous, model get downloaded
                best_acc = p
                torch.save(model.state_dict(), "models/foo.pt")
                print(f"  saved, (accuracy on validation dataset: {p:.3f})")

    model.load_state_dict(torch.load("models/foo.pt", map_location=device))
    total, correct = crnn_algorithm.test_model_debug(model, test_loader, device)
    print(f"correct plates: {total}, correct symbols: {correct}")


if __name__ == "__main__":
    main()