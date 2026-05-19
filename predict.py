import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from colors_to_value import parse_resistor

#from os import putenv
#putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

IMAGE_PATH = "dataset/real/images/IMG_0983.jpg"
BANDNET_WEIGHTS = "bandnet.pt"

USE_DETECTRON = True

OUT_H, OUT_W = 96, 320
MAX_BANDS = 6

COLORS = ["black","brown","red","orange","yellow","green","blue","violet","gray","white","gold","silver"]
PAD = "none"
CLASSES = COLORS + [PAD]

print(f"Using cuda: {torch.cuda.is_available()}")

class BandNet(nn.Module):
    def __init__(self, num_classes=len(CLASSES), max_bands=MAX_BANDS):
        super().__init__()
        self.max_bands = max_bands
        self.num_classes = num_classes

        backbone = models.resnet18(weights=None) 
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(512, max_bands * num_classes)

    def forward(self, x):
        f = self.backbone(x) 
        logits = self.head(f).view(-1, self.max_bands, self.num_classes)
        return logits

def resize_letterbox(img_bgr, out_h=OUT_H, out_w=OUT_W):
    h, w = img_bgr.shape[:2]
    scale = min(out_w / max(w,1), out_h / max(h,1))
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def crop_from_bbox(im_bgr, xyxy, pad=0.15):
    h, w = im_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw; x2 += pad * bw
    y1 -= pad * bh; y2 += pad * bh
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(w - 1, x2)); y2 = int(min(h - 1, y2))
    return im_bgr[y1:y2, x1:y2 if False else x2].copy()  # keep copy

def decode_sequence(probs_1xMxC):
    # probs: [M,C]
    ids = probs_1xMxC.argmax(dim=-1).cpu().tolist()
    conf = probs_1xMxC.max(dim=-1).values.cpu().tolist()

    bands = []
    for i, cid in enumerate(ids):
        name = CLASSES[int(cid)]
        if name == PAD:
            break
        bands.append((name, float(conf[i])))
    return bands

def predict_flip_robust(model, crop_bgr, device):
    crop = resize_letterbox(crop_bgr, OUT_H, OUT_W)
    crop_flip = crop[:, ::-1, :].copy()

    def run_one(img_bgr):
        x = torch.from_numpy(img_bgr).permute(2,0,1).float().unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)            # [1,M,C]
            probs = F.softmax(logits, dim=-1)[0]  # [M,C]
        return probs, img_bgr

    probs1, _ = run_one(crop)
    probs2, _ = run_one(crop_flip)

    seq1 = decode_sequence(probs1)
    seq2 = decode_sequence(probs2)
    seq2 = list(reversed(seq2))  # unflip label order

    s1 = np.mean([c for _,c in seq1]) if seq1 else 0.0
    s2 = np.mean([c for _,c in seq2]) if seq2 else 0.0
    best = seq1 if s1 >= s2 else seq2
    return best, crop


def get_detectron_predictor():
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("resistors_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def detect_bboxes_xyxy(predictor, im_bgr, score_thresh=0.6, max_dets=50):
    """
    Returns a list of (bbox_xyxy, score) sorted by score descending.
    bbox_xyxy is [x1,y1,x2,y2] floats in image pixels.
    """
    outputs = predictor(im_bgr)
    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        return []

    boxes = inst.pred_boxes.tensor.numpy()   # [N,4]
    scores = inst.scores.numpy()             # [N]

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.shape[0] == 0:
        return []

    # sort by score
    order = np.argsort(-scores)
    boxes = boxes[order][:max_dets]
    scores = scores[order][:max_dets]

    return [(boxes[i].tolist(), float(scores[i])) for i in range(len(scores))]



def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Could not find image: {IMAGE_PATH}")
    if not os.path.exists(BANDNET_WEIGHTS):
        raise FileNotFoundError(f"Could not find weights: {BANDNET_WEIGHTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BandNet().to(device)
    state = torch.load(BANDNET_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()

    im = cv2.imread(IMAGE_PATH)
    if im is None:
        raise RuntimeError(f"cv2.imread failed on {IMAGE_PATH}")

    bbox = None
    det_score = None
    crop = im

    predictor = get_detectron_predictor()
    detections = detect_bboxes_xyxy(predictor, im, score_thresh=0.7, max_dets=25)

    if not detections:
        print("No resistors detected; falling back to full image as one crop.")
        detections = [([0, 0, im.shape[1]-1, im.shape[0]-1], 1.0)]


    all_preds = []  # list of dicts with bbox + bands

    for k, (bbox, det_score) in enumerate(detections):
        crop = crop_from_bbox(im, bbox, pad=0.15)
        pred_bands, crop_show = predict_flip_robust(model, crop, device)

        all_preds.append({
            "bbox_xyxy": bbox,
            "det_score": det_score,
            "bands": pred_bands,   # list of (name,conf)
        })

    vis = im.copy()

    for j, p in enumerate(all_preds):
        x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

        #label = ",".join([n for n,_ in p["bands"]]) if p["bands"] else "no-bands"

        cv2.putText(vis, f"{parse_resistor([n for n,_ in p["bands"]], unicode=False)}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.title(f"Detected resistors: {len(all_preds)}")
    plt.tight_layout()
    plt.savefig("band_pred_multi.png", dpi=200)
    print("Saved visualization to band_pred_multi.png")

if __name__ == "__main__":
    main()
