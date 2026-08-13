import os, torch

import torchvision
from PIL import Image
from torchvision import transforms

# same units (cell used, so anchor should be devided by 13)
def get_iou_centered(box1, box2):
    intersection = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
    area = box1[0]*box1[1] + box2[0]*box2[1]

    return intersection / ( area  - intersection + 1e-6)

# images names must be sorted before usage
def garbage_names_clean(all_names, labels_dir_path):
    good_names = []
    for name in all_names:
        name_path = os.path.join(labels_dir_path, os.path.splitext(name)[0] + '.txt')
        if os.path.exists(name_path) and os.path.getsize(name_path) > 0:
            good_names.append(name)

    print(f"kept {len(good_names)} / {len(all_names)} images with good labels")
    return good_names

def get_boxes_from_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            if not line.split():
                continue
            cls, x, y, w, h = map(float, line.split()) # what map is for ?
            boxes.append([cls, x, y, w, h])
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
    return boxes

# target(boxes): lines * [class, x, y, w, h] - tensor (N, 5). encoded GT: (13, 13, 5, 25)
def encode_target(boxes, anchors):

    gt = torch.zeros(13, 13, 5, 25)
    # n = boxes.shape[0]
    for class_id, x, y, w, h in boxes:

        class_id = int(class_id)
        cx, cy = min(int(13 * x), 12), min(int(13 * y), 12)  # row/col indexes of cell where object is
        gx, gy = 13 * x - cx, 13 * y - cy  # image units -> cell units
        gw, gh = 13 * w, 13 * h # image units -> cell units
        box_uncentered = [gw, gh]

        best_iou, best_idx = -1, 0
        for idx, anchor in enumerate(anchors):
            new_iou = get_iou_centered(box_uncentered, anchor)
            if new_iou > best_iou:
                best_iou = new_iou
                best_idx = idx

        if gt[cy, cx, best_idx, 4] == 1:
            continue

        gt[cy, cx, best_idx, 0] = gx
        gt[cy, cx, best_idx, 1] = gy
        gt[cy, cx, best_idx, 2] = gw
        gt[cy, cx, best_idx, 3] = gh
        gt[cy, cx, best_idx, 4] = 1
        gt[cy, cx, best_idx, 5 + class_id] = 1

    return gt

class YOLOv2Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path, labels_dir_path, anchors):
        self.anchors = anchors
        self.labels_dir_path = labels_dir_path
        self.img_dir_path = img_dir_path

        all_names = sorted(os.listdir(img_dir_path))
        self.good_names = garbage_names_clean(all_names, self.labels_dir_path)

        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.good_names)

    def __getitem__(self, idx):
        name = self.good_names[idx]
        img_path = os.path.join(self.img_dir_path, name)
        label_path = os.path.join(self.labels_dir_path, os.path.splitext(name)[0] + '.txt')

        # image resizing and converting ? and what 'transform' means
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        boxes = get_boxes_from_label(label_path)
        gt = encode_target(boxes, self.anchors) # (B, 3, 416, 416)

        return image, gt

def load_one_image(image_path, size=416, device="cpu"):

    img = Image.open(image_path).convert("RGB")
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
    ])
    x = tf(img) # (3, 416, 416)
    return x.unsqueeze(0).to(device) # (1, 3, 416, 416)

# saves current state of training. Evaluate after each epoch.
def save_model_state(model, optimizer, epoch, scheduler, anchors, save_path, name):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "anchors": anchors,
    }, f"{save_path}/{name}.pt")
    print(f"saved {name} model: epoch: {epoch}")


# loads all model training data. Used for resuming training
def load_model_state(model, optimizer, scheduler, load_path, device):
    checkpoint = torch.load(f"{load_path}.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    anchors = checkpoint["anchors"]
    return model, optimizer, scheduler, anchors, epoch

def save_model_weights(model, epoch, save_path, name):
    torch.save(model.state_dict(), f"{save_path}/{name}_epoch_{epoch}.pt")
    print(f"saved {save_path}/{name}/epoch_{epoch}.pt weights")
