import torch, cv2, sys, os
from ultralytics import YOLO

from src.models import crnn_model, yolov2_model
from src.train_architecture import crnn_train_architecture
from src.eval import yolov2_eval_architecture
from src.data_loaders import yolov2_dataload

# global varibles, so functions see them. (They are not downloaded every time function calls them)
yolo8n_detector = YOLO("weights/yolo/yolo8n_fine_tuned.pt")  # pretrained od_yolo on COCO dataset
recognizer = crnn_model.CRNN()  # CRNN class (has forvard method in it)
recognizer.load_state_dict(torch.load("weights/ocr/crnn_best.pt", map_location="cpu"))
recognizer.eval()  # sets training=False flag so that Dropout and BatchNorm2d would not affect models_test forward pass


def read_plate_path(image_path):
    img = cv2.imread(image_path)  # np.array from image
    results = yolo8n_detector(image_path, conf=0.3)  # runs yolo detector
    #                                       ^ confidense that result is correct (output neuron weight)
    out = []
    for r in results:
        for box in r.boxes:  # gives list of corner coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            crop = img[y1:y2, x1:x2]  # numpy slicing
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # like imread but for image (not path)
            gray = cv2.resize(gray, (128, 32))

            t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # torch tensor for ocr model
            with torch.no_grad():
                text = crnn_train_architecture.decode_image(recognizer(t))[0]

            out.append(((x1, y1, x2, y2), text))

    return out

def draw_box_8n_img(frame):  # the same function but with np.array not path
    img = frame
    results = yolo8n_detector(img, conf=0.5)

    out = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 32))

            t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                text = crnn_train_architecture.decode_image(recognizer(t))[0]

            out.append(((x1, y1, x2, y2), text))

    return out


def draw_img_8n(image_path, out_path="results/test.png"):
    img = cv2.imread(image_path)

    for (x1, y1, x2, y2), text in read_plate_path(image_path):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_coordinates = y1 - 10 if y1 > 20 else y2 + 25
        cv2.putText(img, text, (x1, text_coordinates), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    img = cv2.imwrite(out_path, img)
    return img


def draw_video_8n(video_in_path, video_out_path="results/test.mp4"):
    cap = cv2.VideoCapture(video_in_path)  # opens video (np.array is not made, just opens video)
    fps = cap.get(cv2.CAP_PROP_FPS)  # finds video fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # makes video
    writer = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*'mp4v'), # tells to use mp4v codec - which will be saved in mp4 container
        fps,
        (width, height)
    )  # opens file to make video
    while True:
        ok, frame = cap.read()  # gives sequence of (bool, np.array of frame/image)
        if not ok:
            break  # means the end of video

        for (x1, y1, x2, y2), text in draw_box_8n_img(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # the same as in draw_plates
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()  # closes input video
    writer.release()  # closes made video



def decode_plate_v2(img_bgr, model, anchors, conf_threshold=0.4, iou_threshold=0.5, device="cpu"):
    model.eval()

    # img = cv2.imread(image_path)
    x = yolov2_dataload.get_in_tensor_for_dec(img_bgr, 416, device) # (1,3,416,416)
    with torch.no_grad():
        prediction = model(x) # (1,13,13,5,25)
    prediction = prediction[0] # (13,13,5,25)
    # anchors ?
    detected_boxes = yolov2_eval_architecture.get_detected_boxes(
        prediction, anchors, conf_threshold=conf_threshold, iou_boxes_threshold=iou_threshold).cpu() # (N, 6) cell-units

    original_img = img_bgr
    H, W = original_img.shape[:2]
    scale_x = W / 13.0
    scale_y = H / 13.0
    detected_boxes[:, 0] *= scale_x # x1
    detected_boxes[:, 1] *= scale_y # y1
    detected_boxes[:, 2] *= scale_x # x2
    detected_boxes[:, 3] *= scale_y # y2
    out = []
    for box in detected_boxes:
        x1, y1, x2, y2 = box[:4].round().int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = original_img[y1:y2, x1:x2]  # numpy slicing
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # like imread but for image (not path)
        gray = cv2.resize(gray, (128, 32))

        t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # torch tensor for ocr model
        with torch.no_grad():
            text = crnn_train_architecture.decode_image(recognizer(t))[0]

        out.append(((x1, y1, x2, y2), text))

    return out

def decode_plate_v2_path(path, *args, **kwargs):
    return decode_plate_v2(cv2.imread(path), *args, **kwargs)

def decode_plate_v2_frame(frame, *args, **kwargs):
    return decode_plate_v2(frame, *args, **kwargs)

def draw_img_v2(image_path, out_path, conf_threshold, iou_threshold, device="cpu"):
    checkpoint = torch.load("weights/yolo/yolov2_best.pt", map_location=device)
    yolov2_detector = yolov2_model.YOLOv2().to(device)
    yolov2_detector.load_state_dict(checkpoint["state_dict"])
    anchors = checkpoint["anchors"].to(device)
    yolov2_detector.eval()

    img = cv2.imread(image_path)

    for ((x1, y1, x2, y2), text) in decode_plate_v2_path(image_path, yolov2_detector, anchors, conf_threshold, iou_threshold, device):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_coordinates = y1 - 10 if y1 > 20 else y2 + 25
        cv2.putText(img, text, (x1, text_coordinates), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    img = cv2.imwrite(out_path, img)
    return img


def draw_video_v2(video_in_path, video_out_path, conf_threshold, iou_threshold, device="cpu"):
    checkpoint = torch.load("weights/yolo/yolov2_best.pt", map_location=device)
    yolov2_detector = yolov2_model.YOLOv2().to(device)
    yolov2_detector.load_state_dict(checkpoint["state_dict"])
    anchors = checkpoint["anchors"].to(device)
    yolov2_detector.eval()

    cap = cv2.VideoCapture(video_in_path)  # opens video (np.array is not made, just opens video)
    fps = cap.get(cv2.CAP_PROP_FPS)  # finds video fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # makes video
    writer = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*'mp4v'), # tells to use mp4v codec - which will be saved in mp4 container
        fps,
        (width, height)
    )  # opens file to make video
    while True:
        ok, frame = cap.read()  # gives sequence of (bool, np.array of frame/image)
        if not ok:
            break  # means the end of the video

        for (x1, y1, x2, y2), text in decode_plate_v2_frame(frame, yolov2_detector, anchors, conf_threshold, iou_threshold, device):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # the same as in draw_plates
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()  # closes input video
    writer.release()  # closes made video



IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".webm"}

def procces(in_path):
    extension = os.path.splitext(in_path)[1].lower() # splits extension
    base = os.path.splitext(in_path)[0] # splits path

    if extension in IMAGE_EXTS:
        out_path = base + "_out.png"
        draw_img_8n(in_path, out_path)
    elif extension in VIDEO_EXTS:
        out_path = base + "_out.mp4"
        draw_video_8n(in_path, out_path)
    else:
        print("Unsupported file type")

    print(f"Done path: {out_path}")

def main():
    in_path = sys.argv[1]
    procces(in_path)
    # image_path ="test_input_files/corona_image.jpg"
    # out_path = "test_output_files/corona_image.png"
    # draw_plates_recognition(image_path, out_path)

    # video_in_path = "test_input_files/corona.mov"
    # video_out_path = "test_output_files/corona_2.mp4"
    # video_pipeline(video_in_path, video_out_path)


if __name__ == "__main__":
    main()
