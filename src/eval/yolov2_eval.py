import os, torch, time

from src.train_architecture import yolov2_train_architecture
from src.models import yolov2_model
from src.data_loaders import yolov2_dataload
from src.eval import yolov2_eval_architecture


def eval_one_model(load_path, save_path, device):

    os.makedirs(save_path, exist_ok=True)
    print("using", device)

    model = yolov2_model.YOLOv2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = yolov2_train_architecture.get_warmup_schedualer(optimizer)

    model, optimizer, scheduler, anchors, epoch = yolov2_dataload.load_model_state(model, optimizer, scheduler, load_path, device)

    detection_criterion = yolov2_train_architecture.YOLOv2Loss(anchors).to(device)

    img_val_path = "data/dataset/od_nomeroff/images/val"
    labels_val_path = "data/dataset/od_nomeroff/labels/val"
    img_test_path = "data/dataset/od_nomeroff/images/test"
    labels_test_path = "data/dataset/od_nomeroff/labels/test"
    anchors_cpu = anchors.cpu()
    dataset_val = yolov2_dataload.YOLOv2Dataset  (img_val_path,   labels_val_path,   anchors_cpu)
    dataset_test = yolov2_dataload.YOLOv2Dataset (img_test_path,  labels_test_path,  anchors_cpu)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4,persistent_workers=True, pin_memory=True)

    start = time.perf_counter()
    print("calculating validation loss...")
    val_loss = yolov2_eval_architecture.val_loss(model, val_loader, detection_criterion, device)
    end = time.perf_counter()
    elapsed = end - start
    print(f"val loss: {val_loss}, time val_loss: {elapsed:.3f}s.")

    print(f"calculating validation mAP...")
    start = time.perf_counter()
    metrics_val = yolov2_eval_architecture.val_map(model,anchors, img_val_path, labels_val_path, device)
    metrics_test = yolov2_eval_architecture.val_map(model,anchors, img_test_path, labels_test_path, device)
    end = time.perf_counter()
    elapsed = end - start
    print(f"took {elapsed:.3f} seconds")

    with open(save_path + "/val_metrics.csv", "a") as f:
        f.write("epoch_{}, val_loss{:.4f}, mAP@0.5: {:.4f},  mAP@0.75: {:.4f},  mAP@[.5:.95]: {:.4f}\n"
                "mAP@0.5 test: {:.4f},  mAP@0.75 test: {:.4f},  mAP@[.5:.95] test: {:.4f}\n".format(
            epoch, val_loss, metrics_val["mAP_50"], metrics_val["mAP_75"], metrics_val["mAP_5095"],
                               metrics_test["mAP_50"], metrics_test["mAP_75"], metrics_test["mAP_5095"]))

def main():
    load_path, save_path, device = "models_test/yolov2_second/best_epoch_6", "models_test/yolov2_second", torch.device("mps")
    eval_one_model(load_path, save_path, device)

if __name__ == "__main__":
    main()
