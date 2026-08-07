# ANPR — Automatic Number-Plate Recognition using PyTorch

**Detects licence plate and recognises text. Both YOLO(object-detection) and CRNN(OCR) are implemented and trained from scratch using PyTorch.**

## Demo
![Demo](assets/demo.gif)

## Features
 - **YOLOv1 built from scratch** - full model/train architecture and loss-function implemented from original paper, then traied to detect licence plates .
 - **CRNN OCR built from scratch** - CNN + RNN + CTC architecture implemented, then traied to recognise text.
 - **End-to-end inference** - full pipeline for images and video, !!!
 - **Two detection models** - fine-tuned YOLO8n for more precise detections (can be used optionally) 

## Architecture

```mermaid
flowchart LR
    A[Input<br/>image / video] --> B[Plate Detection<br/>YOLO]
    B --> C[Crop plate region]
    C --> D[OCR<br/>CRNN]
    D --> E[Recognized text<br/>e.g. KK568S]
```

### Components
 1. Object-detection
    - **YOLOv1 from scratch** - Trained, evaluated and tested on dataset.
    - **YOLO8n fine-tuned** - pretrained on COCO dataset. Fine-tuned for plate-detection for more precise outcome.
 2. OCR
    - **CRNN** - CNN(feature extractor), RNN(sequence model), CTC(loss function). Implemented using PyTorch. Trained and tested to recognise text on cropped licence plates.

 **Datasets**:  
- YOLOv1 training, YOLO8n fine-tuning: https://nomeroff.net.ua/datasets/
- CRNN training: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset

## Results

## Installation
**Project uses MPS (Metal Performance Shaders) for Mac. If you use NVIDIA then change all 'device=mps' to 'device=cuda'**

```bash
# 1. Clone the repo
git clone https://github.com/dmalynyak/ANPR-plate-recognition
cd ANPR

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Model weights
Two weights are included:  
 - **OCR:** weights/ocr/crnn_best.pt
 - **YOLO8n fine-tuned:** weights/yolo/yolo8n_fine_tuned.pt


## Usage
(doesnt work yet)
1. **Training:**
```bash
  python src/models/crnn_train.py save_path    # OCR training
  python src/models/yolov1_train.py save_path  # YOLOv1 training
```
2. **Inference:**  
Procceses image or video. Gives final output as shown in demo.  
file.ext can be eather image or video.
```bash
    python pipeline.py path/file.ext #saves file path/file_out.ext 
```

