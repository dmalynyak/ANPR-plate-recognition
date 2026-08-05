import torch
from torch.utils.data import DataLoader
import parcing, yolov1_algorithm

def main():
    device = torch.device("mps")
    model = yolov1_algorithm.YOLOv1().to(device)
    model.load_state_dict(torch.load("models/yolov1/first_19.pt", map_location=device))
    model.eval()

    dataset_test = parcing.YoloDataset("dataset/od_nomeroff/images/test",
                                       "dataset/od_nomeroff/labels/test")
    test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)

    for ct in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        acc, p, r, f1 = yolov1_algorithm.test_model( model, test_loader, device, conf_threshold=ct, iou_threshold=0.5)
        print(f"conf_thr={ct}: precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    main()