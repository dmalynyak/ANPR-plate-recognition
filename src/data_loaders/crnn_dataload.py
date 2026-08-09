import os, torch, random, cv2
import numpy as np

from src import parsing


# dataset parsing, returns np.array of pictures + np.array of their names
def load_dataset_kaggle(folder_path, width, heigh):
    X_list = []
    Y_list = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # choose only images
            continue

        filepath = os.path.join(folder_path, filename)
        image_gray, name = parsing.img_parcing_kaggle(filepath, width, heigh)

        X_list.append(image_gray)
        Y_list.append(name)

    return X_list, Y_list


# makes image randomly different. So model does not memorize dataset, but truly learns
def dataset_expansion(img):  # (32,128)
    if random.random() < 0.5:  # 50/50% chance
        a = random.uniform(-5, 5)  # random rotation (-5%, 5%) - just a number
        M = cv2.getRotationMatrix2D((64, 16), a, 1)  # returns rotation matrix(cos -sin sin cos)
        img = cv2.warpAffine(img, M, (128, 32), borderValue=int(img.mean()))  # applies rotation to img. fills borders with avarage colour
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3),0)  # applies Blur to image - convolution(center - 0.25, edges - 0.125, corners - 0.0625.)
    if random.random() < 0.5:
        img = np.clip(img * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)  # changes brightnes from 70% to 130%
    return img


class CRNNDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, expansion=True):
        self.X, self.Y, self.expansion = X, Y, expansion

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        img = self.X[i]  # (32,128)
        if self.expansion:
            img = dataset_expansion(img)  # hardcoded dataset_expansion into dataset loader for torch.utils.data.DataLoader(dataset, batch_size)
        t = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return t, self.Y[i]