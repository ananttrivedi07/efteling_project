import cv2
import os
from pathlib import Path

class TrashFrameExtractor:
    def __init__(self, output_dir, min_contour_area=5000, target_size=(224, 224), warmup_frames=30):
        """
        Initializes the robust video frame extractor pipeline.
        
        :param output_dir: Path object or string directory where uniform crops will be saved.
        :param min_contour_area: Minimum pixel area to consider an object 'trash'.
        :param target_size: Tuple indicating the exact (width, height) to resize crops to.
        :param warmup_frames: Number of initial frames to allow the MOG2 model to stabilize.
        """
        self.output_dir = Path(output_dir)
        self.min_contour_area = min_contour_area
        self.target_size = target_size
        self.warmup_frames = warmup_frames
        
        # Ensure the destination directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Morphological kernels defined once to protect system memory
        self.clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.fuse_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

    def process_single_video(self, video_path):
        """Processes a single video file, filters out initial noise, and saves standardized crops."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        # Fresh background model per video clip to isolate local environment lighting
        backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Generate Motion Mask
            fgMask = backSub.apply(frame)
            
            # UPGRADE: MOG2 Warm-up Phase
            # Skips evaluation while the model learns what the static background looks like
            if frame_count < self.warmup_frames:
                frame_count += 1
                continue
            
            # 2. Morphological Processing (Only active AFTER warm-up phase complete)
            # Clears tiny flickering pixels, then aggressively melts fragmented PET segments together
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.clean_kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, self.fuse_kernel)
            
            # 3. Structural Contour Extraction
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 4. Dynamic 'Zoom Out' Padding (Adds 50% extra breathing room around the object)
                    pad_x = int(w * 0.5) 
                    pad_y = int(h * 0.5)
                    
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(frame.shape[1], x + w + pad_x)
                    y2 = min(frame.shape[0], y + h + pad_y)
                    
                    # Target slice extraction
                    trash_crop = frame[y1:y2, x1:x2]
                    
                    if trash_crop.size == 0:
                        continue
                    
                    # 5. Standardization: Convert to exact model input size (224x224)
                    resized_crop = cv2.resize(trash_crop, self.target_size, interpolation=cv2.INTER_AREA)
                    
                    # 6. File Serialization using a non-overlapping naming syntax
                    save_name = f"{video_path.stem}_frame_{frame_count:04d}.jpg"
                    save_path = self.output_dir / save_name
                    
                    cv2.imwrite(str(save_path), resized_crop)
                    saved_count += 1
                    break  # Maximum of 1 isolated object export per frame
                    
            frame_count += 1

        cap.release()
        print(f"   [SUCCESS] {video_path.name} -> Evaluated: {frame_count} frames | Exported: {saved_count} true trash images.")
        return saved_count

    def extract_from_directory(self, input_dir, extensions=("*.mp4", "*.avi", "*.mov")):
        """Traverses a target directory folder and automatically batch processes all valid video formats."""
        input_dir = Path(input_dir)
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(ext))
            
        if not video_files:
            print(f"[WARN] No compatible video files discovered inside: {input_dir}")
            return
            
        print(f"--- Starting Bulk Uniform Video Extraction Pipeline ---")
        print(f"Target Configuration: Dimensions={self.target_size} | Warmup Threshold={self.warmup_frames} frames")
        print(f"Found {len(video_files)} target source clips in: '{input_dir.name}'\n")
        
        total_extracted = 0
        for video_file in video_files:
            total_extracted += self.process_single_video(video_file)
            
        print(f"\n--- Batch Pipeline Complete ---")
        print(f"Total Database-Ready Images Added to Dataset: {total_extracted}")
        
        
if __name__ == "__main__":
    # Base path relative to where this execution script is saved
    parent = Path(__file__).parent
    
    VIDEOS_DIR = parent / "data" / "video_clips"
    OUTPUT_DIR = parent / "data" / "extracted_trash_frames"
    
    # Initialize the structured extraction engine
    extractor = TrashFrameExtractor(
        output_dir=OUTPUT_DIR,
        min_contour_area=5000,
        target_size=(224, 224),
        warmup_frames=30  # Ignores the first 1 second of static setup noise
    )
    
    # Process the entire video directory sweep
    extractor.extract_from_directory(input_dir=VIDEOS_DIR)