import torch
from loguru import logger
from ultralytics import YOLO
import cv2

import crnn_algorithm

# global varibles, so functions see them. (They are not downloaded every time function calls them)
yolo_detector = YOLO("models/yolo_third/weights/best.pt")  # pretrained od_yolo on COCO dataset
recognizer = crnn_algorithm.CRNN()  # CRNN class (has forvard method in it)
recognizer.load_state_dict(torch.load("models/1_0_nomeroff_expansion_30.pt", map_location="cpu"))
recognizer.eval()  # sets training=False flag so that Dropout and BatchNorm2d would not affect models forward pass


def read_plate_path(image_path):
    img = cv2.imread(image_path)  # np.array from image
    results = yolo_detector(image_path, conf=0.3)  # runs yolo detector
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
                text = crnn_algorithm.decode_image(recognizer(t))[0]

            out.append(((x1, y1, x2, y2), text))

    return out


def read_plate_frame(frame):  # the same function but with np.array not path
    img = frame
    results = yolo_detector(img, conf=0.5)

    out = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 32))

            t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                text = crnn_algorithm.decode_image(recognizer(t))[0]

            out.append(((x1, y1, x2, y2), text))

    return out


def draw_plates_recognition(image_path, out_path="results/test.png"):
    img = cv2.imread(image_path)

    for (x1, y1, x2, y2), text in read_plate_path(image_path):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_coordinates = y1 - 10 if y1 > 20 else y2 + 25
        cv2.putText(img, text, (x1, text_coordinates), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    img = cv2.imwrite(out_path, img)
    if not img:
        logger.error(f"failed to write {out_path}kkk")
    return img


def video_pipeline(video_in_path, video_out_path="results/test.mp4"):
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

        for (x1, y1, x2, y2), text in read_plate_frame(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # the same as in draw_plates
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()  # closes input video
    writer.release()  # closes made video


def main():
    image_path ="test_input_files/corona_image.jpg"
    out_path = "test_output_files/corona_image.png"
    draw_plates_recognition(image_path, out_path)

    # video_in_path = "test_input_files/corona.mov"
    # video_out_path = "test_output_files/corona.mp4"
    # video_pipeline(video_in_path, video_out_path)


if __name__ == "__main__":
    main()
