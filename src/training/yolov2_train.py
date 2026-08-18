import os, torch, time, argparse

from src.eval import yolov2_eval_architecture
from src.train_architecture import yolov2_train_architecture
from src.models import yolov2_model
from src.data_loaders import yolov2_dataload


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
            raise RuntimeError("testCUDA requested but not available on this machine")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device '{name}'")


def train_logic(model, optimizer, scheduler_state, anchors, detection_criterion, epoch_start, epoch_end, save_path, device):

    img_train_path = "data/dataset/od_nomeroff/images/train"
    labels_train_path = "data/dataset/od_nomeroff/labels/train"
    img_val_path = "data/dataset/od_nomeroff/images/val"
    labels_val_path = "data/dataset/od_nomeroff/labels/val"
    img_test_path = "data/dataset/od_nomeroff/images/test"
    labels_test_path = "data/dataset/od_nomeroff/labels/test"
    dataset_train = yolov2_dataload.YOLOv2Dataset(img_train_path, labels_train_path, anchors)
    dataset_val = yolov2_dataload.YOLOv2Dataset  (img_val_path,   labels_val_path,   anchors)
    dataset_test = yolov2_dataload.YOLOv2Dataset (img_test_path,  labels_test_path,  anchors)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4,persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4,persistent_workers=True, pin_memory=True)

    steps_per_epoch = len(train_loader)
    scheduler = yolov2_train_architecture.build_scheduler(optimizer, epochs=50, steps_per_epoch=steps_per_epoch)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    best_map = 0.0
    for epoch in range(epoch_start, epoch_end):


        train_loss = yolov2_train_architecture.train_epoch(model, train_loader, detection_criterion, optimizer,
                                                           scheduler, save_path, device, epoch)
        start = time.perf_counter()
        print("calculating validation loss...")
        val_loss = yolov2_eval_architecture.val_loss(model, val_loader, detection_criterion, device)
        end = time.perf_counter()
        elapsed = end - start
        with open(save_path + "/val_metrics.csv", "a") as f:
            f.write("epoch_{}, train_loss: {:.4f}, val_loss: {:.4f}\n".format(
                epoch,   train_loss,           val_loss, ))
        print(f"epoch: {epoch} train loss: {train_loss:.3f}, val loss: {val_loss}, time val_loss: {elapsed:.3f}s.")


        if epoch % 5 == 0:
            print(f"epoch {epoch}, calculating validation mAP...")
            start = time.perf_counter()
            metrics = yolov2_eval_architecture.val_map(model, anchors, img_val_path, labels_val_path, device)

            with open(save_path + "/val_metrics.csv", "a") as f:
                f.write("epoch_{}, val_loss: {:.4f}, mAP@0.5: {:.4f},  mAP@0.75: {:.4f},  mAP@[.5:.95]: {:.4f}\n".format(
                    epoch,   val_loss,        metrics["mAP_50"], metrics["mAP_75"], metrics["mAP_5095"]))

            end = time.perf_counter()
            elapsed = end - start
            print(f"took {elapsed:.3f} seconds")
            if metrics["mAP_5095"] > best_map:
                best_map = metrics["mAP_5095"]
                yolov2_dataload.save_model_state(model, optimizer, epoch, scheduler, anchors, save_path=save_path, name=f"best_mAP.pt")
                yolov2_dataload.save_model_weights(model, anchors, save_path=save_path, name='best_weights.pt')

        yolov2_dataload.save_model_state(model, optimizer, epoch, scheduler, anchors, save_path=save_path, name=f"last.pt")

    checkpoint = torch.load(f"{save_path}/best_mAP.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = yolov2_eval_architecture.val_map(model, anchors, img_test_path, labels_test_path, device)
    with open(save_path + "/val_metrics.csv", "a") as f:
        f.write("test_mAP@0.5: {:.4f},  test_mAP@0.75: {:.4f},  test_mAP@[.5:.95]: {:.4f}\n".format(
            test_metrics["mAP_50"], test_metrics["mAP_75"], test_metrics["mAP_5095"]))


def train_from_zero(save_path, device):

    os.makedirs(save_path, exist_ok=True)
    print("using", device)

    start = time.perf_counter()
    anchors = yolov2_train_architecture.get_anchors("data/dataset/od_nomeroff/labels/train")
    end = time.perf_counter()
    elapsed = end - start
    print(f"anchors are defined:\n"
          f"{anchors}\n"
          f"time: {elapsed:.3f}s")

    model = yolov2_model.YOLOv2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    detection_criterion = yolov2_train_architecture.YOLOv2Loss(anchors).to(device)

    scheduler_state = None
    train_logic(model, optimizer, scheduler_state, anchors, detection_criterion, epoch_start=0, epoch_end=50 + 1, save_path=save_path, device=device)


def train_resume(load_path, save_path, device):
    os.makedirs(save_path, exist_ok=True)

    checkpoint = torch.load(f"{load_path}", map_location=device)
    anchors = checkpoint["anchors"].cpu()
    epoch_start = checkpoint["epoch"]

    model = yolov2_model.YOLOv2().to(device)
    model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])

    detection_criterion = yolov2_train_architecture.YOLOv2Loss(anchors).to(device)
    print(f"Resuming training from epoch {epoch_start} with anchors:\n{anchors}")

    train_logic(model, optimizer, checkpoint["scheduler"], anchors, detection_criterion,
                epoch_start + 1, epoch_end=50 + 1, save_path=save_path, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-path", help="path to folder where model states will be saved")
    parser.add_argument("--resume", default=None, help="checkpoint path with .pt to resume from")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.resume:
        train_resume(args.resume, args.save_path, device)
    else:
        train_from_zero(args.save_path, device)

if __name__ == "__main__":
    main()
