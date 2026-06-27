import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from pathlib import Path

class TrashClassifierBatch:
    def __init__(self, model_path, num_classes=2, confidence_threshold=90.0):
        """
        Initializes the batch inference pipeline with a confidence safety net.
        
        :param model_path: Path to the saved weights file.
        :param num_classes: Number of output targets.
        :param confidence_threshold: Float percentage (0-100). PET predictions below this 
                                     value are automatically downgraded to 'OTHER'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing inference on device: {self.device}")
        print(f"Safety Gate Active: PET requires >= {confidence_threshold}% confidence.")
        
        self.model = resnet18(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.confidence_threshold = confidence_threshold

    def classify_image(self, img_path):
        """Runs single inference and applies the strict confidence threshold rule."""
        try:
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probabilities, 1)
                
            raw_pred = pred.item()          # 0 for OTHER, 1 for PET
            confidence_pct = conf.item() * 100
            
            # --- THE SAFETY GATE LOGIC ---
            # If the model thinks it's PET (1) but is guessing blindly, force it to OTHER (0)
            if raw_pred == 1 and confidence_pct >= self.confidence_threshold:
                final_label = "PET"
            else:
                final_label = "OTHER"
                
            return final_label, confidence_pct
            
        except Exception as e:
            print(f"   [ERROR] Skipping broken image {img_path.name}: {e}")
            return None, None

    def process_directory_incremental(self, input_dir, summary_output_dir, save_every=10):
        """Scans a directory, runs thresholded classification, and saves visual summaries."""
        input_dir = Path(input_dir)
        summary_output_dir = Path(summary_output_dir)
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_extensions = ('.jpg', '.jpeg', '.png')
        img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
        
        if not img_paths:
            print(f"[WARN] No valid images found in directory: {input_dir}")
            return
        
        print(f"--- Running Thresholded Batch Inference ({len(img_paths)} total images) ---")
        
        current_batch_results = []
        batch_counter = 1
        
        for idx, path in enumerate(img_paths, 1):
            label, confidence = self.classify_image(path)
            if label:
                current_batch_results.append({
                    "path": path,
                    "label": label,
                    "confidence": confidence
                })
            
            if idx % save_every == 0 or idx == len(img_paths):
                if current_batch_results:
                    output_file_path = summary_output_dir / f"summary_batch_{batch_counter:03d}.jpg"
                    self._save_summary_strip(current_batch_results, output_file_path, num_samples=5)
                    print(f"[BATCH {batch_counter:03d}] Handled up to image #{idx}/{len(img_paths)}.")
                    
                    current_batch_results = []
                    batch_counter += 1
        
        print(f"\n--- Process Complete ---")

    def _save_summary_strip(self, results, output_path, num_samples=5):
        """Selects random samples from the current batch pool and saves them side-by-side."""
        sample_count = min(len(results), num_samples)
        sampled_results = random.sample(results, sample_count)
        
        fig, axes = plt.subplots(1, sample_count, figsize=(15, 4), layout='constrained')
        
        if sample_count == 1:
            axes = [axes]
            
        for i, item in enumerate(sampled_results):
            img = Image.open(item["path"])
            axes[i].imshow(img)
            
            title_color = "green" if item["label"] == "PET" else "red"
            axes[i].set_title(f"{item['label']}\n({item['confidence']:.1f}%)", color=title_color, fontsize=12, weight='bold')
            axes[i].axis('off')
            
        plt.savefig(output_path, dpi=120)
        plt.close()
        
        
if __name__ == "__main__":
    parent = Path(__file__).parent
    
    # Paths Configuration
    MODEL_WEIGHTS = parent / "model_resnet18.pt"
    FRAMES_DIR = parent / "data" / "extracted_trash_frames"
    SUMMARY_DIR = parent / "data" / "batch_summaries"
    
    # 1. Instantiate the classifier
    classifier = TrashClassifierBatch(model_path=MODEL_WEIGHTS)
    
    # 2. Run with save_every=10
    classifier.process_directory_incremental(
        input_dir=FRAMES_DIR,
        summary_output_dir=SUMMARY_DIR,
        save_every=10  # Change this to 50 or 100 later if your folder fills up too fast!
    )