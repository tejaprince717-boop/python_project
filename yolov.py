# ============================
# Setup (Colab/Jupyter friendly)
# ============================
import os, sys, time, glob, math
import cv2
import torch
import numpy as np

# Install ultralytics if missing
try:
    from ultralytics import YOLO
except ImportError:
    !pip install -q ultralytics
    from ultralytics import YOLO

# Colab-safe display (optional)
COLAB = False
try:
    from google.colab.patches import cv2_imshow
    COLAB = True
except Exception:
    pass

# ============================
# Config
# ============================
MODEL_WEIGHTS   = "yolov8x-seg.pt"   # accuracy first; switch to s/n for speed
CONF_THRESHOLD  = 0.45               # raise for fewer false positives
IOU_THRESHOLD   = 0.5
IMG_SIZE        = 640
DEVICE          = 0 if torch.cuda.is_available() else "cpu"  # use GPU if available
SAVE_PREVIEW    = True               # save a JPG preview of the first annotated frame

# Optional: focus on common road classes only to boost precision.
# None = all classes; else provide list of class IDs from COCO:
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7, traffic light=9, stop sign=11
FOCUS_CLASSES = [0,1,2,3,5,7,9,11]

# ============================
# Utilities
# ============================
def readable_time(seconds: float) -> str:
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    return f"{mm:02d}:{ss:02d}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# ============================
# Core: process a single video
# ============================
def process_video(
    input_path: str,
    output_path: str,
    model: YOLO,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=IMG_SIZE,
    classes=FOCUS_CLASSES,
    preview=True,
):
    assert os.path.isfile(input_path), f"Video not found: {input_path}"
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {input_path}")
        return None

    # Video properties
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")

    ensure_dir(os.path.dirname(output_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"▶  Processing: {os.path.basename(input_path)} | {width}x{height} @ {fps:.1f} FPS | {frames} frames")
    print(f"    Saving to : {output_path}")

    # Stats
    frame_idx = 0
    t0 = time.time()
    fps_smooth = None
    first_frame_saved = False
    obj_counter = 0  # total detections across frames (rough indicator)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        tic = time.time()

        # YOLOv8 segmentation predict on raw frame (numpy array)
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=DEVICE,
            classes=classes,   # None for all
            verbose=False
        )

        # Annotated frame (BGR)
        annotated = results[0].plot()

        # Count detections for logging (boxes)
        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            obj_counter += len(results[0].boxes)

        # FPS smoothing
        inst_fps = 1.0 / max(1e-6, time.time() - tic)
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

        # Overlay FPS
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(annotated)

        # Optional: save or show a preview of first annotated frame
        if preview and not first_frame_saved and frame_idx == 1:
            preview_path = os.path.splitext(output_path)[0] + "_preview.jpg"
            cv2.imwrite(preview_path, annotated)
            print(f"    📸 Saved preview: {preview_path}")
            if COLAB:
                cv2_imshow(annotated)
            first_frame_saved = True

        # (Optional) print progress every N seconds
        if frame_idx % max(1, int(fps)) == 0:
            elapsed = time.time() - t0
            done_ratio = frame_idx / max(1, frames)
            eta = (elapsed / max(1e-6, done_ratio)) - elapsed if frames > 0 else 0
            print(f"    [{frame_idx:>6}/{frames if frames>0 else '?':>6}] "
                  f"{done_ratio*100:5.1f}% | "
                  f"avg FPS ~{(frame_idx/elapsed):.1f} | "
                  f"ETA {readable_time(eta)}", end="\r")

    cap.release()
    out.release()
    total_time = time.time() - t0
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print()
    print(f"✅ Done: {os.path.basename(input_path)} | {frame_idx} frames in {readable_time(total_time)} | avg FPS {avg_fps:.1f}")
    print(f"   Total detections (approx.): {obj_counter}")
    return {
        "video": input_path,
        "frames": frame_idx,
        "time_sec": total_time,
        "avg_fps": avg_fps,
        "total_dets": obj_counter,
        "output": output_path
    }

# ============================
# Process all videos in folder
# ============================
def process_dataset(
    input_dir: str,
    output_dir: str,
    pattern="*.mp4",
    **kwargs
):
    ensure_dir(output_dir)
    model = YOLO(MODEL_WEIGHTS)  # load once
    # (Optional) fuse layers for a tiny speed boost
    try:
        model.fuse()
    except Exception:
        pass

    # Collect videos by common extensions
    exts = [pattern] if "" in pattern else [f".{pattern}"]
    vids = []
    for ext in exts + [".avi", ".mov", ".mkv", ".MTS", ".MP4", ".MOV"]:
        vids.extend(glob.glob(os.path.join(input_dir, ext)))

    assert len(vids) > 0, f"No videos found in {input_dir}. Put .mp4/.avi files there."

    summaries = []
    for v in vids:
        base = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(output_dir, base + "_yolov8xseg.mp4")
        summary = process_video(v, out_path, model=model, **kwargs)
        if summary:
            summaries.append(summary)

    print("\n=== Summary ===")
    for s in summaries:
        print(f"- {os.path.basename(s['video'])}: {s['frames']} frames | {s['avg_fps']:.1f} FPS | out={s['output']}")
    return summaries

