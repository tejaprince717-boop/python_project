!git clone https://github.com/Dt-Pham/Advanced-Lane-Lines.git
%cd Advanced-Lane-Lines/



import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import tensorflow as tf

# ===== LANENET INTEGRATION =====
class LaneNetModel:
    """LaneNet deep learning model integration (optional)."""
    def __init__(self, model_path='lanenet_model.h5'):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Try to load LaneNet model (Keras .h5) if present."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ LaneNet model loaded successfully")
            except Exception as e:
                print("⚠️ Failed to load LaneNet model:", e)
                self.model = None
        else:
            print("ℹ️ LaneNet model not found. Using traditional methods")
            self.model = None

    def preprocess(self, image_bgr):
        """Preprocess BGR image for LaneNet (expects BGR -> RGB -> resize -> normalize)."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 256))
        img_norm = img_resized.astype(np.float32) / 127.5 - 1.0
        return np.expand_dims(img_norm, axis=0)

    def predict(self, image_bgr):
        """Predict lanes using LaneNet. Returns binary_mask (H x W uint8 0/255) and instance embedding or None."""
        if self.model is None:
            return None, None

        input_tensor = self.preprocess(image_bgr)
        try:
            prediction = self.model.predict(input_tensor, verbose=0)
        except Exception as e:
            print("⚠️ LaneNet prediction failed:", e)
            return None, None

        # Accept either list [binary, embedding] or single output
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            binary = prediction[0][0]
            embedding = prediction[1][0] if len(prediction) > 1 else None
        else:
            # Single output assumed to be binary mask
            binary = prediction[0]
            embedding = None

        # Convert to uint8 binary mask (match original image size: we'll resize later if necessary)
        binary_mask = (binary > 0.5).astype(np.uint8) * 255
        # binary_mask shape is (H, W) for 512x256
        return binary_mask, embedding

    def cluster_lanes(self, binary_mask, embedding):
        """Optional: cluster instance embeddings into different lane labels (returns colored mask)."""
        if embedding is None:
            return binary_mask

        # Ensure shapes align: embedding expected shape (H, W, C), binary_mask (H, W)
        h, w = binary_mask.shape[:2]
        if embedding.shape[0] != h or embedding.shape[1] != w:
            # resize embedding to mask size
            embedding = cv2.resize(embedding, (w, h), interpolation=cv2.INTER_LINEAR)

        y_idx, x_idx = np.where(binary_mask > 0)
        if len(y_idx) == 0:
            return binary_mask

        features = embedding[y_idx, x_idx]  # (N, C)
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            labels = dbscan.fit_predict(features)
        except Exception as e:
            print("⚠️ DBSCAN failed:", e)
            return binary_mask

        colored_mask = np.zeros_like(binary_mask)
        for (y, x), label in zip(zip(y_idx, x_idx), labels):
            if label >= 0:
                colored_mask[y, x] = (label + 1) * 50  # arbitrary coloring scale
        return colored_mask
        y_idx, x_idx = np.where(binary_mask > 0)
        if len(x_idx) > 0:
            fit = np.polyfit(y_idx, x_idx, 2)  # quadratic fit
            y_vals = np.linspace(0, binary_mask.shape[0]-1, binary_mask.shape[0])
            x_vals = np.polyval(fit, y_vals).astype(np.int32)
        for i in range(len(y_vals)-1):
              cv2.line(img_out, (x_vals[i], int(y_vals[i])), (x_vals[i+1], int(y_vals[i+1])), (0,0,255), 4)


# ===== CUSTOM CLASSES =====
class CameraCalibration:
    """Simplified camera calibration placeholder (identity / example intrinsics)."""
    def __init__(self):
        # Example intrinsic; replace with real calibration for production
        self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def undistort(self, img_bgr):
        # For now, this will just return the original image (no distortion correction)
        try:
            return cv2.undistort(img_bgr, self.camera_matrix, self.dist_coeffs)
        except Exception:
            return img_bgr


class Thresholding:
    """Classical thresholding (HLS + Sobel) to get a binary lane mask (0 or 255)."""
    def forward(self, img_bgr):
        # Convert to HLS and grayscale (OpenCV uses BGR input)
        hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:,:,1]
        white_mask = cv2.inRange(img_bgr, (200,200,200), (255,255,255))
        combined_binary = np.zeros_like(s_channel)
        combined_binary[((s_channel>170) & (s_channel<=255)) | ((l_channel>200) & (l_channel<=255)) | (white_mask>0)] = 255
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1.0
        scaled_sobel = np.uint8(255 * abs_sobelx / max_val)

        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

        sx_binary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        combined_binary = np.zeros_like(sx_binary, dtype=np.uint8)
        combined_binary[(s_binary == 1) | (sx_binary == 1)] = 255  # use 0/255 for mask
        return combined_binary


class PerspectiveTransformation:
    """Perspective transform (bird's-eye) - user-defined source/destination points."""
    def __init__(self, src_points=None, dst_points=None):
        # Default points (example) - you should tune these for your camera
        if src_points is None:
            self.src_points = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
        else:
            self.src_points = np.float32(src_points)

        if dst_points is None:
            self.dst_points = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])
        else:
            self.dst_points = np.float32(dst_points)

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def forward(self, img_bgr):
        h, w = img_bgr.shape[:2]
        return cv2.warpPerspective(img_bgr, self.M, (w, h))

    def backward(self, img_bgr):
        h, w = img_bgr.shape[:2]
        return cv2.warpPerspective(img_bgr, self.M_inv, (w, h))


class LaneLines:
    """Lane line detection and visualization."""
    def __init__(self, lanenet_model_path='lanenet_model.h5'):
        self.lanenet = LaneNetModel(model_path=lanenet_model_path)

    def forward(self, img_bgr):
        """
        Return:
          vis_img_bgr: visualization image (same shape as input)
          binary_mask: single-channel mask (0 or 255) aligned with vis_img
        """
        # Try LaneNet first (if loaded)
        binary_mask, instance_embedding = self.lanenet.predict(img_bgr)

        if binary_mask is not None:
            # If laneNet returns mask of size (H,W) for 512x256, resize to input image
            mask_h, mask_w = binary_mask.shape[:2]
            h, w = img_bgr.shape[:2]
            if (mask_h, mask_w) != (h, w):
                binary_mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                binary_mask_resized = binary_mask

            # If instance embeddings present, create colored clustered mask
            if instance_embedding is not None:
                colored_mask = self.lanenet.cluster_lanes(binary_mask_resized, instance_embedding)
                colored_mask = colored_mask.astype(np.uint8)
                vis = np.dstack([colored_mask, colored_mask, colored_mask])
            else:
                vis = np.dstack([binary_mask_resized, binary_mask_resized, binary_mask_resized])

            return vis, binary_mask_resized

        # Fallback to traditional method
        binary_img = Thresholding().forward(img_bgr)
        binary_img = binary_img.astype(np.uint8)

        # Visualize the binary image as color (green channel)
        vis = np.zeros_like(img_bgr)
        vis[:, :, 1] = binary_img  # green channel

        return vis, binary_img

    def draw_lines_on_image(self, base_img_bgr, binary_mask):
        """Draw lane lines from binary mask onto base image and return combined image."""
        img_out = base_img_bgr.copy()
        h, w = binary_mask.shape[:2]

        # Use probabilistic Hough to detect line segments
        # Prepare edges (binary_mask is 0/255)
        edges = binary_mask.copy()
        # optional: perform Canny to get better edges
        edges = cv2.Canny(edges, 50, 150)

        # HoughLinesP params - tune for your scenario
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=int(w*0.1), maxLineGap=int(w*0.05))
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 3)  # red lines

        return img_out

    def plot(self, img_bgr, binary_mask=None):
        """Overlay mask and draw lane lines and method text."""
        out = img_bgr.copy()

        # overlay mask with transparency
        if binary_mask is not None:
            colored_mask = np.zeros_like(out)
            colored_mask[:, :, 1] = binary_mask  # green where mask present
            out = cv2.addWeighted(out, 1.0, colored_mask, 0.5, 0)

            # Draw Hough lines on top
            out = self.draw_lines_on_image(out, binary_mask)

        # Put method text
        if self.lanenet.model is not None:
            method_text = "LaneNet Detection"
            color = (0, 255, 0)
        else:
            method_text = "Traditional CV Detection"
            color = (0, 255, 255)

        cv2.putText(out, method_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return out


# ===== MAIN APPLICATION CLASS =====
class FindLaneLines:
    """Main lane detection application with LaneNet support."""
    def __init__(self, lanenet_model_path='lanenet_model.h5'):
        self.calibration = CameraCalibration()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines(lanenet_model_path)
        print("✅ Lane detection system initialized")

    def forward(self, img_bgr):
        """Full pipeline: undistort -> bird's eye -> lane detection -> reproject -> overlay."""
        original = img_bgr.copy()

        # undistort (no-op by default)
        undistorted = self.calibration.undistort(img_bgr)

        # perspective transform to bird's eye
        bird = self.transform.forward(undistorted)

        # detect lanes (vis_bird is visualization in bird-eye space, mask is single-channel)
        vis_bird, binary_mask_bird = self.lanelines.forward(bird)

        # reproject visualization and mask back to original view
        vis_back = self.transform.backward(vis_bird)
        mask_back = self.transform.backward(binary_mask_bird)

        # ensure mask is single channel 0/255 and same size
        if len(mask_back.shape) == 3:
            mask_back = cv2.cvtColor(mask_back, cv2.COLOR_BGR2GRAY)
        mask_back = (mask_back > 127).astype(np.uint8) * 255

        # Blend visualization with original
        blended = cv2.addWeighted(original, 0.7, vis_back, 0.3, 0)

        # Draw refined Hough lines and overlay mask on blended image
        final = self.lanelines.plot(blended, mask_back)

        return final

    def process_image(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"❌ Input image not found: {input_path}")
            return

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print(f"❌ Failed to read image: {input_path}")
            return

        out_img = self.forward(img_bgr)
        cv2.imwrite(output_path, out_img)
        print(f"✅ Image processed and saved to: {output_path}")

    def process_video(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"❌ Input video not found: {input_path}")
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        print("⏳ Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.forward(frame)
            out.write(processed)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"📈 Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"✅ Processed {frame_count} frames. Output: {output_path}")


# ===== MAIN FUNCTION =====
def main():
    print("🚗 Advanced Lane Detection")
    print("=" * 50)

    detector = FindLaneLines(lanenet_model_path='lanenet_model.h5')

    print("\nChoose an option:")
    print("1. Process image")
    print("2. Process video")
    print("3. Exit")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        input_path = input("Input image path: ").strip()
        output_path = input("Output image path [output.jpg]: ").strip() or "output.jpg"
        detector.process_image(input_path, output_path)

    elif choice == '2':
        input_path = input("Input video path: ").strip()
        output_path = input("Output video path [output.mp4]: ").strip() or "output.mp4"
        detector.process_video(input_path, output_path)

    elif choice == '3':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
