import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import random
import torch
import json
from PIL import Image
from torchvision import transforms


##############################  CRNN ##################################

# ---------- ocr_kaggle ---------------

# image paring, returns 0-1 grayscale np.array(size1, size2) and name
def img_parcing_kaggle(path, width, heigh):
    basename = os.path.basename(path)  # TK012D.png
    name = os.path.splitext(basename)[0]  # TK012D
    name = name.split('_')[0].upper()  # for name '9JAG121_1.jpg' to work fine
    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 0-1 grayscale np.array
    image_gray = cv2.resize(image_gray, (width, heigh))

    return image_gray, name


# dataset parsing, returns np.array of pictures + np.array of their names
def load_dataset_kaggle(folder_path, width, heigh):
    X_list = []
    Y_list = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # choose only images
            continue

        filepath = os.path.join(folder_path, filename)
        image_gray, name = img_parcing_kaggle(filepath, width, heigh)

        X_list.append(image_gray)
        Y_list.append(name)

    return X_list, Y_list


# ---------- ocr_nomeroff ---------------


def load_dataset_nomeroff(images_folder, descriptions_folder, width, heigh):
    X_list, Y_list = [], []
    for description_name in os.listdir(descriptions_folder):  # iterates throw descriptions names (file name)
        if not description_name.endswith(".json"):
            continue

        with open(os.path.join(descriptions_folder,
                               description_name)) as f:  # opens description (file) inside descriptions_folder
            annotation = json.load(f)  # loads json annotation for image (ann name = img name) as np.array

        # path to image (file) inside images_folder with the same name as annotation (file)
        img_path = os.path.join(images_folder, description_name.replace(".json", ".png"))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:  # cv2.imread as error gives None
            continue

        X_list.append(cv2.resize(img, (width, heigh)))
        Y_list.append(annotation["description"].upper())  # json structure- "description": number plate

    return X_list, Y_list


def load_cache_nomeroff(images_folder, descriptions_folder, width, heigh, cache_name):
    x_path, y_path = f"{cache_name}_X.npy", f"{cache_name}_Y.npy"

    if os.path.exists(x_path):
        print(f"loading cache {cache_name}")
        X = list(np.load(x_path))
        Y = list(np.load(y_path))
    else:
        print(f"parsing {images_folder} and saving to cache")
        X, Y = load_dataset_nomeroff(images_folder, descriptions_folder, width, heigh)
        os.makedirs("cache", exist_ok=True)
        np.save(x_path, np.array(X))
        np.save(y_path, np.array(Y))

    return X, Y


# ----------- dataset loading -------------- (for torch.utils.data.DataLoader)


# makes image randomly different. So model does not memorize dataset, but truly learns
def dataset_expansion(img):  # (32,128)
    if random.random() < 0.5:  # 50/50% chance
        a = random.uniform(-5, 5)  # random rotation (-5%, 5%) - just a number
        M = cv2.getRotationMatrix2D((64, 16), a, 1)  # returns rotation matrix(cos -sin sin cos)
        img = cv2.warpAffine(img, M, (128, 32),
                             borderValue=int(img.mean()))  # applies rotation to img. fills borders with avarage colour
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3),
                               0)  # applies Blur to image - convolution(center - 0.25, edges - 0.125, corners - 0.0625.)
    if random.random() < 0.5:
        img = np.clip(img * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)  # changes brightnes from 70% to 130%
    return img


class PlateDataset(Dataset):
    def __init__(self, X, Y, expansion=True):
        self.X, self.Y, self.expansion = X, Y, expansion

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        img = self.X[i]  # (32,128)
        if self.expansion:
            img = dataset_expansion(
                img)  # hardcoded dataset_expansion into dataset loader for torch.utils.data.DataLoader(dataset, batch_size)
        t = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return t, self.Y[i]


##############################  OD ##################################


def via_to_yolo(json_path, images_dir_path, out_dir_path):
    if os.path.exists(out_dir_path):  # if directory with regions exists, then skip all function
        print(f"dataset length: {len(os.listdir(out_dir_path))}")
        return
    os.makedirs(out_dir_path, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    os.makedirs(out_dir_path, exist_ok=True)
    meta = data["_via_img_metadata"]  # skips information on top of JSON file

    for _, img_data in meta.items():  # structure of json
        fname = img_data["filename"]

        try:
            with Image.open(os.path.join(images_dir_path, fname)) as im:
                W, H = im.size  # dimensions of the image
        except (FileNotFoundError, OSError):  # checks if image with this path exists
            continue

        lines = []  # stores final YOLO format data

        for region in img_data["regions"]:
            sa = region["shape_attributes"]
            xs, ys = sa["all_points_x"], sa["all_points_y"]

            x0, x1 = min(xs), max(xs)  # polygon (bounding box coordinates)
            y0, y1 = min(ys), max(ys)

            xc = (x0 + x1) / 2 / W  # center
            yc = (y0 + y1) / 2 / H
            bw = (x1 - x0) / W  # width, height (YOLO format)
            bh = (y1 - y0) / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if lines:  # creates file img_label.txt (YOLO format)
            out_name = os.path.splitext(fname)[0] + ".txt"
            with open(os.path.join(out_dir_path, out_name), "w") as f:
                f.write("\n".join(lines))
# target(7,7,25): for target[i][j]  [0:20] class one-hot   [20] objectness   [21:25] x,y,w,h
# def encode_yolo_target(raw_boxes): # list of (int(object_class), x, y, w, h)
#     target = np.zeros((7, 7, 25), dtype=np.float32)
#     for box in raw_boxes:
#         class_id, x, y, w, h = box
#         target[int(x // (1 / 7))][int(y // (1 / 7))][class_id] = 1  # class index
#         target[int(x // (1 / 7))][int(y // (1 / 7))][20] = 1 # objectness
#         target[int(x // (1 / 7))][int(y // (1 / 7))][20 + 1] = x # x coordinate
#         target[int(x // (1 / 7))][int(y // (1 / 7))][20 + 2] = y # y coordinate
#         target[int(x // (1 / 7))][int(y // (1 / 7))][20 + 3] = w # w width
#         target[int(x // (1 / 7))][int(y // (1 / 7))][20 + 4] = h # h heigh
#
#     return target # (7, 7, 25)

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


class YoloDataset(Dataset):
    def __init__(self, image_dir, labels_dir):
        self.img_dir_path = image_dir
        all_files = sorted(os.listdir(image_dir)) # returns sorted NAMES (not actual files)
        self.img_files = [ name for name in all_files
            if os.path.exists(os.path.join(labels_dir, os.path.splitext(name)[0] + '.txt'))
        ]
        # self.img_files = self.img_files[:100]
        print(f"kept {len(self.img_files)} / {len(all_files)} images with labels")
        self.labels_dir_path = labels_dir


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): # returns (image, target) = [(3, 448, 448), (7, 7, 25)]
        name = self.img_files[idx] # returns file name
        img_path = os.path.join(self.img_dir_path, name)
        image = Image.open(img_path).convert('RGB')

        base = os.path.splitext(name)[0]
        label_path = os.path.join(self.labels_dir_path, base + '.txt')
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
        image = transform(image)


        return image, boxes_encoded
