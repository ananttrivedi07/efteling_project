import cv2
import os
import shutil
from pathlib import Path

class ManualAnnotator:
    def __init__(self, input_dir, output_base_dir):
        """
        Initializes the manual sorting tool.
        
        :param input_dir: Directory containing the unclassified images.
        :param output_base_dir: Where the sorted folders will be created.
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create output class directories
        self.pet_dir = self.output_base_dir / "PET"
        self.other_dir = self.output_base_dir / "OTHER"
        
        self.pet_dir.mkdir(parents=True, exist_ok=True)
        self.other_dir.mkdir(parents=True, exist_ok=True)
        
        # Cross-platform keycode mapping for OpenCV
        self.KEY_LEFT = [2424832, 65361, 63234, ord('a')]   # OTHER
        self.KEY_RIGHT = [2555904, 65363, 63235, ord('d')]  # PET
        self.KEY_QUIT = [ord('q'), 27]                      # 'q' or ESC

    def run_annotation(self):
        """Starts the interactive UI loop for sorting videos in bulk."""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        img_paths = [p for p in self.input_dir.iterdir() if p.suffix.lower() in valid_extensions]
        
        if not img_paths:
            print(f"[INFO] No images found in {self.input_dir}. All done!")
            return
            
        # --- NEW: Group all frame paths by their source video name ---
        video_groups = {}
        for path in img_paths:
            video_name = path.name.split('_frame_')[0]
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(path)
            
        print(f"--- Video-Level Data Annotator Started ---")
        print(f"Total Videos to sort: {len(video_groups)} (comprising {len(img_paths)} images)")
        print("CONTROLS:")
        print("  [ RIGHT ARROW ] or [ D ] -> Classify WHOLE VIDEO as PET")
        print("  [ LEFT ARROW ]  or [ A ] -> Classify WHOLE VIDEO as OTHER (Residual)")
        print("  [ Q ] or [ ESC ]         -> Save & Quit")
        print("-" * 40)
        
        videos_sorted = 0
        images_sorted = 0
        
        for video_name, frames in video_groups.items():
            # Sort frames and pick the middle one to show (best chance of a clear view)
            frames.sort()
            middle_idx = len(frames) // 2
            rep_frame_path = frames[middle_idx]
            
            img = cv2.imread(str(rep_frame_path))
            if img is None:
                continue
                
            display_img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
            
            # Add visual overlay instructions
            cv2.putText(display_img, "<- OTHER (Left) | (Right) PET ->", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_img, f"Video: {video_name} ({len(frames)} frames)", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_img, "Press 'Q' to Quit", (50, 580), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Data Annotator", display_img)
            
            key = cv2.waitKeyEx(0) 
            
            if key in self.KEY_QUIT:
                print("\n[INFO] Annotation paused. You can resume later.")
                break
                
            elif key in self.KEY_RIGHT:
                # Move ALL frames in this video to PET
                for f_path in frames:
                    dest = self.pet_dir / f_path.name
                    shutil.move(str(f_path), str(dest))
                print(f"[{videos_sorted + 1}/{len(video_groups)}] -> Labeled Video: {video_name} as PET")
                videos_sorted += 1
                images_sorted += len(frames)
                
            elif key in self.KEY_LEFT:
                # Move ALL frames in this video to OTHER
                for f_path in frames:
                    dest = self.other_dir / f_path.name
                    shutil.move(str(f_path), str(dest))
                print(f"[{videos_sorted + 1}/{len(video_groups)}] -> Labeled Video: {video_name} as OTHER")
                videos_sorted += 1
                images_sorted += len(frames)
                
            else:
                print("[WARN] Invalid key pressed. Skipping video... (It remains in the input folder)")

        cv2.destroyAllWindows()
        print(f"\n--- Session Complete ---")
        print(f"Successfully labeled {videos_sorted} videos ({images_sorted} total images) for fine-tuning.")
        
if __name__ == "__main__":
    parent = Path(__file__).parent

    UNCLASSIFIED_DIR = parent / "data" / "extracted_trash_frames"
    SORTED_DATASET_DIR = parent / "data" / "fine_tuning_dataset"

    annotator = ManualAnnotator(
        input_dir=UNCLASSIFIED_DIR, 
        output_base_dir=SORTED_DATASET_DIR
    )

    annotator.run_annotation()