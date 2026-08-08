from pathlib import Path
import torch

# same units (cell used, so anchor should be devided by 13)
def get_iou_centered(box1, box2):
    intersection = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
    area = box1[0]*box1[1] + box2[0]*box2[1]

    return intersection / ( area  - intersection + 1e-6)

# same units
def get_iou_uncentered(box0, box1):
    b0x1, b0y1 = box0[0:1] - box0[2:3]/2, box0[1:2] - box0[3:4]/2
    b0x2, b0y2 = box0[0:1] + box0[2:3]/2, box0[1:2] + box0[3:4]/2
    b1x1, b1y1 = box1[0:1] - box1[2:3]/2, box1[1:2] - box1[3:4]/2
    b1x2, b1y2 = box1[0:1] + box1[2:3]/2, box1[1:2] + box1[3:4]/2

    intersection = (torch.min(b0x2, b1x2) - torch.max(b0x1, b1x1)).clamp(0) * (torch.min(b0y2, b1y2) - torch.max(b0y1, b1y1)).clamp(0)
    area = box0[2:3]*box0[3:4] + box1[2:3]*box1[3:4]

    return intersection / (area - intersection + 1e-6)

def get_anchors(labels_path):

    folder_path = Path(labels_path)
    boxes = []
    for file_path in folder_path.iterdir():
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                boxes.append([float(parts[-2]), float(parts[-1]), -1])

    anchors = 13 * torch.rand(5, 2)

    while True:
        flag = False
        for box in boxes:
            iou = -1
            for idx, anchor in enumerate(anchors):
                new_iou = get_iou_centered(box, anchor/13)
                if new_iou > iou:
                    if box[2] != idx:
                        flag = True
                    box[2] = idx
                    iou = new_iou
        if not flag:
            break

        anchors_new = torch.zeros(5, 3)
        for box in boxes:
            anchor_idx = box[2]
            anchors_new[anchor_idx][0] += box[0]
            anchors_new[anchor_idx][1] += box[1]
            anchors_new[anchor_idx][2] += 1 # number of boxes where anchor is responsible
        for idx in range(5):
            if anchors_new[idx, 2] == 0:
                continue
            for j in range(2):
                anchors[idx, j] = 13 * anchors_new[idx, j] / anchors_new[idx, 2] # makes cell-units for anchor/13 to work properly

    return anchors


# pred: (13, 13, 5, 25) decoded: (13, 13, 5, 25)
def decode_prediction(pred, anchors):
    torch.no_grad()

    dec = torch.clone(pred) # makes independant copy
    S = 13

    cx = torch.arange(S).view(1, S, 1)  # (1, 13, 1) broadcasts to (13, 13, 5) where (13, i, 5) = i
    cy = torch.arange(S).view(S, 1, 1)  # (13, 1, 1)
    pw = anchors[:, 0].view(1, 1, 5)  # broadcasts to (13, 13, 5)
    ph = anchors[:, 1].view(1, 1, 5)

    dec[..., 0] = torch.sigmoid(pred[..., 0]) + cx  # bx = σ(tx) + col index
    dec[..., 1] = torch.sigmoid(pred[..., 1]) + cy
    dec[..., 2] = pw * torch.exp(pred[..., 2]) # bw
    dec[..., 3] = ph * torch.exp(pred[..., 3]) # bh
    dec[..., 4] = torch.sigmoid(pred[..., 4]) # confidence
    dec[..., 5:25] = torch.softmax(pred[..., 5:25], dim = -1)

    return dec

# for parcing. target(boxes): lines * [class, x, y, w, h] - tensor (N, 5). encoded GT: (13, 13, 5, 25)
def encode_target(boxes, anchors):

    gt = torch.zeros(13, 13, 5, 25)
    n = boxes.shape[0]
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




