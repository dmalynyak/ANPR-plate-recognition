import torch, time, argparse

from src.train_architecture import yolov1_train_architecture
from src.models import yolov1_model
from src.data_loaders import yolov1_dataload


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

    dataset_train = yolov1_dataload.YOLOv1Dataset("data/dataset/od_nomeroff/images/train", "data/dataset/od_nomeroff/labels/train")
    dataset_val = yolov1_dataload.YOLOv1Dataset("data/dataset/od_nomeroff/images/val", "data/dataset/od_nomeroff/labels/val")
    dataset_test = yolov1_dataload.YOLOv1Dataset("data/dataset/od_nomeroff/images/test", "data/dataset/od_nomeroff/labels/test")

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)

    model = yolov1_model.YOLOv1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val = float("inf")
    for epoch in range(50):
        start = time.perf_counter()
        train_loss = yolov1_train_architecture.train_epoch(model, train_loader, yolov1_train_architecture.criterion, optimizer, device, epoch)
        val_loss = yolov1_train_architecture.validate(model, val_loader, yolov1_train_architecture.criterion, device)
        accuracy, precision, recall, F1 = yolov1_train_architecture.test_model(model, test_loader, device, conf_threshold=0.4, iou_threshold=0.5)
        end = time.perf_counter()
        elapsed = end - start
        print(f"epoch: {epoch} train loss: {train_loss:.3f} val loss: {val_loss:.3f} time: {elapsed:.3f} \n"
              f"accuracy (bad metric here): {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f}, F1: {F1:.3f}")

        if epoch % 5 == 0:
            if val_loss < best_val:
                torch.save(model.state_dict(), f"{save_path}/best.pt")
                print(f"model saved, epoch: {epoch}")
                with open(f"{save_path}/weights_info.txt", "a") as file:
                    file.write(f"epoch: {epoch}\n"
                               f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, time: {elapsed:.3f} \n"
                               f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, F1: {F1:.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-path", help="path to folder where model states will be saved")
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_logic(args.save_path, device)


if __name__ == "__main__":
    main()