# ===== Super Fast DeepLabV3+ Segmentation =====
import os, time, cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# -------- Config --------
VIDEO_PATH   = "/content/5150590-uhd_3840_2160_30fps.mp4"
OUTPUT_VIDEO = "deeplab22_fast.mp4"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE_WIDTH, RESIZE_HEIGHT = 320, 180   # much smaller = faster
FRAME_SKIP   = 9   # process 1 frame, skip 9
MAX_FRAMES   = 200

# -------- Load model --------
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights).to(DEVICE).eval()
if DEVICE.type == "cuda":
    model = model.half()  # FP16

CLASSES = weights.meta["categories"]
np.random.seed(0)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# Fast preprocessing (no torchvision transforms)
def preprocess_fast(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (520, 520))  # DeepLab expects ~520
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
    return img.half() if DEVICE.type == "cuda" else img

@torch.no_grad()
def segment(frame):
    h, w = frame.shape[:2]
    inp = preprocess_fast(frame)
    out = model(inp)["out"]
    label_map = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    label_map = cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = frame.copy()
    detections = []
    for cls_id in np.unique(label_map):
        if cls_id == 0: continue
        mask = (label_map == cls_id)
        if mask.sum() < 500: continue
        color = COLORS[cls_id]
        overlay[mask] = (overlay[mask]*0.5 + color*0.5).astype(np.uint8)
        detections.append(CLASSES[cls_id])
    return overlay, detections

def run_video(video_path=VIDEO_PATH, out_path=OUTPUT_VIDEO):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (RESIZE_WIDTH, RESIZE_HEIGHT))

    frame_idx, processed = 0, 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % (FRAME_SKIP+1) != 0: continue
        if MAX_FRAMES and processed >= MAX_FRAMES: break

        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        overlay, det = segment(frame)
        out.write(overlay)
        processed += 1
        print(f"Frame {frame_idx}: {det}")

    cap.release(); out.release()
    total_t = time.time() - t0
    print(f"Done {processed} frames in {total_t:.1f}s | FPS ~{processed/(total_t+1e-6):.2f}")

# ---- Run ----
if __name__ == "__main__":
    run_video()
