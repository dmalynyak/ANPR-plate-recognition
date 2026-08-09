import os, torch
import numpy as np

from PIL import Image
from torchvision import transforms


# target(7,7,25): for target[i][j]  [0:20] class one-hot   [20] objectness   [21:25] x,y,w,h
def encode_yolo_target(raw_boxes):
    target = np.zeros((7, 7, 25), dtype=np.float32)
    for class_id, x, y, w, h in raw_boxes:
        col = min(int(x * 7), 6)
        row = min(int(y * 7), 6)
        x_cell = x * 7 - col          # cell relative
        y_cell = y * 7 - row
        target[row][col][class_id] = 1  # class index
        target[row][col][20] = 1 # objectness
        target[row][col][21] = x_cell # x coordinate
        target[row][col][22] = y_cell # y coordinate
        target[row][col][23] = w # w width
        target[row][col][24] = h # h heigh

    return target # (7, 7, 25)


class YOLOv1Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels_dir_path):
        self.img_dir_path = image_dir
        self.labels_dir_path = labels_dir_path

        self.img_files = []
        all_files = sorted(os.listdir(image_dir))  # returns sorted NAMES (not actual files)
        for name in all_files:
            if os.path.exists(os.path.join(labels_dir_path, os.path.splitext(name)[0] + '.txt')):
                self.img_files.append(name)
        # self.img_files = self.img_files[:100]
        print(f"kept {len(self.img_files)} / {len(all_files)} images with good labels")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): # returns (image, target) = [(3, 448, 448), (7, 7, 25)]
        name = self.img_files[idx] # returns file name
        img_path = os.path.join(self.img_dir_path, name)

        label_path = os.path.join(self.labels_dir_path, os.path.splitext(name)[0] + '.txt')
        boxes = [] # these boxes are going to be targets to push to during training
        with open(label_path) as f:
            for line in f:
                object_class, x, y, w, h = map(float, line.split())
                boxes.append((int(object_class), x, y, w, h))
        boxes_encoded = encode_yolo_target(boxes)

        transform = transforms.Compose([
            transforms.Resize((448, 448)),  # normalized x,y,w,h unchanged
            transforms.ToTensor(),  # pixels are [0,1]
        ])
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        return image, boxes_encoded