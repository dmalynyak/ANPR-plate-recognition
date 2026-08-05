import sys

import torch
from torch.utils.data import DataLoader
import time

import parcing
import yolov1_algorithm

def main():
    save_path = sys.argv[1]
    #save_path = "models/yolov1/first"
    device = torch.device("mps")
    print("using", device)

    dataset_train = parcing.YoloDataset("dataset/od_nomeroff/images/train", "dataset/od_nomeroff/labels/train")
    dataset_val = parcing.YoloDataset("dataset/od_nomeroff/images/val", "dataset/od_nomeroff/labels/val")
    dataset_test = parcing.YoloDataset("dataset/od_nomeroff/images/test", "dataset/od_nomeroff/labels/test")

    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4, persistent_workers=True)

    model = yolov1_algorithm.YOLOv1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        start = time.perf_counter()
        train_loss = yolov1_algorithm.train_epoch(model, train_loader, yolov1_algorithm.criterion, optimizer, device)
        val_loss = yolov1_algorithm.validate(model, val_loader, yolov1_algorithm.criterion, device)
        accuracy, precision, recall, F1 = yolov1_algorithm.test_model(model, test_loader, device, conf_threshold=0.7, iou_threshold=0.5)
        end = time.perf_counter()
        elapsed = end - start
        print(f"epoch: {epoch} train loss: {train_loss:.3f} val loss: {val_loss:.3f} time: {elapsed:.3f} \n"
              f"accuracy (bad metric here): {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f}, F1: {F1:.3f}")

        torch.save(model.state_dict(), f"{save_path}_{epoch}.pt")
        print(f"model saved, epoch: {epoch}")
        with open(f"{save_path}_info.txt", "a") as file:
            file.write(f"epoch: {epoch}\n"
                       f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, time: {elapsed:.3f} \n"
                       f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, F1: {F1:.3f}\n")

if __name__ == "__main__":
    main()