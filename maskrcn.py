import cv2
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# -------------------------
# Load model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.to(device).eval()

# COCO classes
COCO_CLASSES = weights.meta["categories"]

# Random color per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def run_inference(frame, threshold=0.5):
    # Convert frame to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(img)[0]

    boxes = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    masks = outputs['masks'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = labels[i]
        color = [int(c) for c in COLORS[cls_id]]

        # Draw bounding box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{COCO_CLASSES[cls_id]} {scores[i]:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw mask (blend with color)
        mask = masks[i, 0] > 0.5
        frame[mask] = (0.5 * frame[mask] + 0.5 * np.array(color)).astype(np.uint8)

    return frame


# -------------------------
# IMAGE EXAMPLE
# -------------------------
image_path = "/content/sample_data/test1.jpg"   # change to your image path
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

result = run_inference(img, threshold=0.6)
cv2.imwrite("output.jpg", result)
print("Saved result -> output.jpg")


# -------------------------
# VIDEO EXAMPLE
# -------------------------
"""
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = run_inference(frame, threshold=0.6)
    out.write(frame)
    cv2.imshow("Mask R-CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
"""