import cv2, os, json
import numpy as np
from PIL import Image


# for CRNN
# ---------- ocr_kaggle ---------------

# image paring, returns 0-1 grayscale np.array(size1, size2) and name
def img_parcing_kaggle(path, width, heigh):
    basename = os.path.basename(path)  # TK012D.png
    name = os.path.splitext(basename)[0]  # TK012D
    name = name.split('_')[0].upper()  # for name '9JAG121_1.jpg' to work fine
    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 0-1 grayscale np.array
    image_gray = cv2.resize(image_gray, (width, heigh))

    return image_gray, name


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
        os.makedirs("../cache", exist_ok=True)
        np.save(x_path, np.array(X))
        np.save(y_path, np.array(Y))

    return X, Y


# for YOLO
# --------------- od_nomeroff ----------------

def via_to_yolo(json_path, images_dir_path, out_dir_path):
    if os.path.exists(out_dir_path):  # if directory with regions exists, then skip all function
        print(f"parced dataset exists, length: {len(os.listdir(out_dir_path))}")
        return
    print(f"dataset parcing...")

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

    print(f"dataset parced, length: {len(os.listdir(out_dir_path))}")
