import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


class VideoTrashClassifier:
    """
    Loads a fine-tuned ResNet18 checkpoint and classifies frames from
    .mp4 videos in a given folder (default: data/synthetic_data).

    NEW: instead of feeding the ResNet the full frame, we run background
    subtraction (MOG2) to find moving blobs (the trash), crop around them
    with padding, and classify each crop. This matches how the model was
    trained (on cropped, mostly-single-object images) much better than
    feeding it a full frame with background/clutter baked in.
    """

    def __init__(
        self,
        model_path="model_resnet18.pt",
        num_classes=2,
        video_dir="data/synthetic_data",
        frame_skip=1,          # 1 = process every frame. Raise if fast objects allow it.
        class_names=None,
        device=None,
        # --- background subtraction / cropping params ---
        bg_history=200,        # frames used to build the background model
        bg_var_threshold=16,   # MOG2 sensitivity (lower = more sensitive)
        bg_detect_shadows=False,
        min_contour_area=800,  # px^2, filters out noise blobs. Tune to your resolution.
        max_contour_area_frac=0.9,  # ignore blobs that are basically the whole frame
        crop_padding_frac=0.15,     # pad the tight bbox by this fraction on each side
        warmup_frames=30,      # frames used purely to prime the background model, not classified
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = Path(video_dir)
        self.frame_skip = max(1, frame_skip)
        self.class_names = class_names or {0: "OTHER", 1: "PET"}

        self.min_contour_area = min_contour_area
        self.max_contour_area_frac = max_contour_area_frac
        self.crop_padding_frac = crop_padding_frac
        self.warmup_frames = warmup_frames
        self._bg_history = bg_history
        self._bg_var_threshold = bg_var_threshold
        self._bg_detect_shadows = bg_detect_shadows

        self.model = resnet18(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    # ------------------------------------------------------------------
    # Background subtraction / candidate region extraction
    # ------------------------------------------------------------------
    def _new_bg_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self._bg_history,
            varThreshold=self._bg_var_threshold,
            detectShadows=self._bg_detect_shadows,
        )

    def _get_candidate_boxes(self, fg_mask, frame_shape):
        """
        Turn a foreground mask into a list of (x1, y1, x2, y2) candidate boxes,
        cleaned up with morphology and filtered by area.
        """
        h, w = frame_shape[:2]
        max_area = self.max_contour_area_frac * (h * w)

        # Clean up noise: erode small speckle, dilate to merge nearby fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)

            # pad the box
            pad_w = int(bw * self.crop_padding_frac)
            pad_h = int(bh * self.crop_padding_frac)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + bw + pad_w)
            y2 = min(h, y + bh + pad_h)
            boxes.append((x1, y1, x2, y2))

        return boxes

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_crop(self, frame_bgr, box):
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return self.class_names[pred.item()], conf.item()

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------
    def process_video(self, video_path, show=False):
        """
        Classify moving-object crops in a single video.
        Returns a list of dicts: {frame, timestamp_s, box, label, confidence}.
        One entry per detected object per processed frame (can be zero or
        multiple entries per frame).
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        bg_subtractor = self._new_bg_subtractor()
        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Always feed the subtractor so the background model stays current,
            # even on frames we skip for classification.
            fg_mask = bg_subtractor.apply(frame)

            is_warmup = frame_idx < self.warmup_frames
            should_classify = (not is_warmup) and (frame_idx % self.frame_skip == 0)

            boxes = []
            if should_classify:
                boxes = self._get_candidate_boxes(fg_mask, frame.shape)

                for box in boxes:
                    out = self._classify_crop(frame, box)
                    if out is None:
                        continue
                    label, conf = out
                    results.append({
                        "frame": frame_idx,
                        "timestamp_s": round(frame_idx / fps, 3),
                        "box": box,
                        "label": label,
                        "confidence": conf,
                    })

            if show:
                vis = frame.copy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                for r in [r for r in results if r["frame"] == frame_idx]:
                    x1, y1, x2, y2 = r["box"]
                    color = (0, 255, 0) if r["label"] == "PET" else (0, 0, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis, f"{r['label']} ({r['confidence']*100:.1f}%)",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                cv2.imshow(video_path.name, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if show:
            cv2.destroyAllWindows()

        return results

    def process_folder(self, show=False, pattern="*.mp4"):
        """
        Classify every video in self.video_dir.
        Returns {video_filename: [detection-level results]}.
        """
        video_files = sorted(self.video_dir.glob(pattern))
        if not video_files:
            print(f"No videos matching '{pattern}' found in {self.video_dir}")
            return {}

        all_results = {}
        for video_path in video_files:
            print(f"Processing {video_path.name} ...")
            all_results[video_path.name] = self.process_video(video_path, show=show)

        return all_results


if __name__ == "__main__":
    classifier = VideoTrashClassifier(
        model_path="model_resnet18.pt",
        video_dir="data/synthetic_data",
        frame_skip=1,            # every frame, since trash moves fast
        min_contour_area=800,    # TUNE: depends on your video resolution / object size
        crop_padding_frac=0.15,
        warmup_frames=30,        # ~1s at 30fps to let MOG2 learn the empty background
    )

    results = classifier.process_folder(show=True)

    # Simple summary: how many detections were classified as PET per video
    for video_name, detections in results.items():
        pet_count = sum(1 for d in detections if d["label"] == "PET")
        print(f"{video_name}: {pet_count}/{len(detections)} detections classified as PET")