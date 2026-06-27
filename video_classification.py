import cv2
import os
import shutil
from pathlib import Path

class VideoAnnotator:
    def __init__(self, input_dir, output_base_dir):
        """
        Initializes the video sorting tool with frame-accurate step controls.
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create output class directories
        self.pet_dir = self.output_base_dir / "PET"
        self.other_dir = self.output_base_dir / "OTHER"
        
        self.pet_dir.mkdir(parents=True, exist_ok=True)
        self.other_dir.mkdir(parents=True, exist_ok=True)
        
        # Cross-platform keycode mapping
        # REMOVED 'a' and 'd' from here so they don't trigger categorization
        self.KEY_LEFT = [2424832, 65361, 63234]              # LEFT ARROW ONLY -> OTHER
        self.KEY_RIGHT = [2555904, 65363, 63235]             # RIGHT ARROW ONLY -> PET
        self.KEY_QUIT = [ord('q'), 27]                       # 'q' or ESC
        self.KEY_SPACE = [32]                                # Spacebar (Pause/Play)
        self.KEY_PREV_FRAME = [ord('a'), ord('A')]           # A -> Frame Backward
        self.KEY_NEXT_FRAME = [ord('d'), ord('D')]           # D -> Frame Forward

    def run_annotation(self):
        """Starts the interactive video player loop with frame-by-frame precision."""
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_paths = [p for p in self.input_dir.iterdir() if p.suffix.lower() in valid_extensions]
        
        if not video_paths:
            print(f"[INFO] No videos found in {self.input_dir}. All done!")
            return
            
        print(f"--- Frame-Accurate Video Annotator Started ---")
        print(f"Total Videos to sort: {len(video_paths)}")
        print("CONTROLS:")
        print("  [ SPACEBAR ]    -> Pause / Play")
        print("  [ D ]           -> Step 1 Frame FORWARD (Pauses video)")
        print("  [ A ]           -> Step 1 Frame BACKWARD (Pauses video)")
        print("  [ RIGHT ARROW ] -> Classify WHOLE VIDEO as PET")
        print("  [ LEFT ARROW ]  -> Classify WHOLE VIDEO as OTHER (Residual)")
        print("  [ Q ] or [ ESC ]-> Save & Quit")
        print("-" * 65)
        
        sorted_count = 0
        
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / video_fps) if video_fps > 0 else 33
            
            decision_made = False
            quit_requested = False
            is_paused = False
            last_valid_frame = None
            
            while not decision_made:
                if not is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video back to frame 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    last_valid_frame = frame.copy()
                else:
                    frame = last_valid_frame.copy()

                if frame is None:
                    continue

                # Note: cap.get returns the index of the NEXT frame to be read
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                display_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
                
                # --- DRAW PROGRESS BAR ---
                bar_x1, bar_y1 = 50, 530
                bar_x2, bar_y2 = 750, 540
                ratio = (current_frame / total_frames) if total_frames > 0 else 0
                progress_width = int((bar_x2 - bar_x1) * ratio)
                
                cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
                cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x1 + progress_width, bar_y2), (255, 255, 0), -1)
                
                # --- HUD OVERLAYS ---
                status_text = "PAUSED" if is_paused else "PLAYING"
                status_color = (0, 0, 255) if is_paused else (0, 255, 0)
                
                cv2.putText(display_frame, f"STATUS: {status_text}", (50, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display_frame, "<- OTHER (Left Arrow) | (Right Arrow) PET ->", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Video: {video_path.name}", (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                cv2.putText(display_frame, f"Frame: {current_frame} / {total_frames}", (50, 515), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "[SPACE] Play/Pause | [A] Frame <- | [D] Frame -> | [Q] Quit", (50, 580), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("Video Annotator", display_frame)
                
                wait_time = 0 if is_paused else delay
                key = cv2.waitKeyEx(wait_time)
                
                # --- KEY CONTROL HANDLERS ---
                if key in self.KEY_SPACE:
                    is_paused = not is_paused
                    
                elif key in self.KEY_NEXT_FRAME:
                    is_paused = True  # Halt video stream
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if ret:
                        last_valid_frame = frame.copy()
                        
                elif key in self.KEY_PREV_FRAME:
                    is_paused = True  # Halt video stream
                    # OpenCV read pointer is currently at current_frame. 
                    # To step 1 frame backward from what we are *seeing*, we jump back 2 steps.
                    target_frame = max(0, current_frame - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    if ret:
                        last_valid_frame = frame.copy()
                    
                elif key in self.KEY_QUIT:
                    quit_requested = True
                    decision_made = True
                    
                elif key in self.KEY_RIGHT:
                    cap.release()
                    dest = self.pet_dir / video_path.name
                    shutil.move(str(video_path), str(dest))
                    print(f"[{sorted_count + 1}/{len(video_paths)}] -> Labeled: PET")
                    sorted_count += 1
                    decision_made = True
                    
                elif key in self.KEY_LEFT:
                    cap.release()
                    dest = self.other_dir / video_path.name
                    shutil.move(str(video_path), str(dest))
                    print(f"[{sorted_count + 1}/{len(video_paths)}] -> Labeled: OTHER")
                    sorted_count += 1
                    decision_made = True

            cap.release()
            if quit_requested:
                print("\n[INFO] Annotation paused. You can resume later.")
                break

        cv2.destroyAllWindows()
        print(f"\n--- Session Complete ---")
        print(f"Successfully labeled {sorted_count} videos.")

if __name__ == "__main__":
    parent = Path(__file__).parent

    UNCLASSIFIED_VIDEOS_DIR = parent / "data" / "video_clips"
    SORTED_VIDEOS_DIR = parent / "data" / "sorted_video_clips"

    annotator = VideoAnnotator(
        input_dir=UNCLASSIFIED_VIDEOS_DIR, 
        output_base_dir=SORTED_VIDEOS_DIR
    )

    annotator.run_annotation()