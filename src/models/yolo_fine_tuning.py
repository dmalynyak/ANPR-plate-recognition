from ultralytics import YOLO


def main(
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 64,
        device: str = 'cpu',
        project: str = 'models_test',
        name: str = 'yolo_01'
):
    parcing.via_to_yolo("dataset/od_nomeroff/ann/train/via_region_data.json", "dataset/od_nomeroff/images/train",
                        "dataset/od_nomeroff/labels/train")
    parcing.via_to_yolo("dataset/od_nomeroff/ann/val/via_region_data.json", "dataset/od_nomeroff/images/val",
                        "dataset/od_nomeroff/labels/val")
    parcing.via_to_yolo("dataset/od_nomeroff/ann/test/via_region_data.json", "dataset/od_nomeroff/images/test",
                        "dataset/od_nomeroff/labels/test")

    model = YOLO('yolov8n.pt')  # pretrained model on COCO

    model.train(
        data='dataset/od_nomeroff/yolo_info.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,  # can be changed to 'mps'
        project=project,
        name=name,
    )


if __name__ == "__main__":
    main()
