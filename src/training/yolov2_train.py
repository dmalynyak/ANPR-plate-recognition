import os, sys, torch, time

from src.train_architecture import yolov2_train_architecture
from src.models import yolov2_model
from src.data_loaders import yolov2_dataload



def train_logic(model, optimizer, scheduler, anchors, detection_criterion, epoch_start, epoch_end, save_path, device):

    img_train_path = "data/dataset/od_nomeroff/images/train"
    labels_train_path = "data/dataset/od_nomeroff/labels/train"
    img_val_path = "data/dataset/od_nomeroff/images/val"
    labels_val_path = "data/dataset/od_nomeroff/labels/val"
    img_test_path = "data/dataset/od_nomeroff/images/train"
    labels_test_path = "data/dataset/od_nomeroff/labels/train"
    dataset_train = yolov2_dataload.YOLOv2Dataset(img_train_path, labels_train_path, anchors)
    dataset_val = yolov2_dataload.YOLOv2Dataset  (img_val_path,   labels_val_path,   anchors)
    dataset_test = yolov2_dataload.YOLOv2Dataset (img_test_path,  labels_test_path,  anchors)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4,persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4,persistent_workers=True, pin_memory=True)

    best_val_loss = float("inf")
    best_map = 0.0
    for epoch in range(epoch_start, epoch_end):

        start = time.perf_counter()
        train_loss = yolov2_train_architecture.train_epoch(model, train_loader, detection_criterion, optimizer,
                                                           scheduler, save_path, device, epoch)
        val_loss = yolov2_train_architecture.val_loss(model, val_loader, detection_criterion, device)
        # yolov2_train_architecture.save_model_state(model, optimizer, epoch, scheduler, save_path=save_path, name='latest')
        end = time.perf_counter()
        elapsed = end - start
        print(f"epoch: {epoch} train loss: {train_loss:.3f}, val loss: {val_loss}, time: {elapsed}")

        if val_loss < best_val_loss:
            yolov2_train_architecture.save_model_state(model, optimizer, epoch, scheduler, anchors, save_path=save_path, name=f"best_loss_epoch{epoch}")

        metrics = yolov2_train_architecture.val_map(model,anchors, img_val_path, labels_val_path)
        if metrics["mAP_5095"] > best_map:
            best_map = metrics["mAP_5095"]
            yolov2_train_architecture.save_model_state(model, optimizer, epoch, scheduler, anchors, save_path=save_path, name=f"best_mAP_epoch{epoch}")

        yolov2_train_architecture.save_model_weights(model, epoch, save_path=save_path, name='weights')


def train_from_zero(save_path):

    os.makedirs(save_path, exist_ok=True)
    device = torch.device("mps")
    print("using", device)

    start = time.perf_counter()
    anchors = yolov2_train_architecture.get_anchors("data/dataset/od_nomeroff/labels/train")
    end = time.perf_counter()
    elapsed = end - start
    print(f"anchors are defined: {anchors}, time: {elapsed:.3f}")

    model = yolov2_model.YOLOv2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = yolov2_train_architecture.get_warmup_schedualer(optimizer)
    detection_criterion = yolov2_train_architecture.YOLOv2Loss(anchors).to(device)

    train_logic(model, optimizer, scheduler, anchors, detection_criterion, epoch_start=0, epoch_end=30, save_path=save_path, device=device)


def train_resume(load_path, save_path):

    os.makedirs(save_path, exist_ok=True)
    device = torch.device("mps")
    print("using", device)

    model = yolov2_model.YOLOv2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = yolov2_train_architecture.get_warmup_schedualer(optimizer)

    model, optimizer, scheduler, anchors, epoch_start = (
        yolov2_train_architecture.load_model_state(model, optimizer, scheduler, load_path, device))

    detection_criterion = yolov2_train_architecture.YOLOv2Loss(anchors).to(device)

    train_logic(model, optimizer, scheduler, anchors,  detection_criterion, epoch_start, epoch_end=30, save_path=save_path, device=device)


def main():
    train_from_zero("models_test/yolov2_third")
    # train_resume("models_test/yolov2_first/best.pt", "models_test/yolov2_first")

if __name__ == "__main__":
    main()