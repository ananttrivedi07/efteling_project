from collections import defaultdict
import os
import json
import math
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

import torchvision.transforms as transforms
import openml_pytorch as opt
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader

class ModelLogger(pl.Callback):
    """ Helper class to log metrics and save models in a systematic way.
        Don't modify this class unless you have a sound reason to do so.

    Attributes:
        output_dir (str) : Directory to save model and metrics (base_dir by default)
        model_name (str) : Name of the model. Should be model_1 for your final model in Question 1.
        use_half (bool)  : Whether to save model in half precision. This reduces file size but is
                           less accurate and not ideal for continued training.
        device (str)     : Device to save the model to (cpu, cuda, mps).
    """
    def __init__(self, model_name="model", test_loader=None, output_dir="./", use_half=False, device="cpu"):
        super().__init__()
        self.output_dir = output_dir
        self.model_name = model_name
        self.test_loader = test_loader
        self.use_half = use_half
        self.device = device
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.current_val_loss = 0.0
        self.current_val_acc = 0.0
        os.makedirs(output_dir, exist_ok=True)

    # Logs the loss and accuracy for each epoch
    def log_epoch(self, train_loss, val_loss, train_acc=None, val_acc=None):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc if train_acc is not None else 0.0)
        self.history["val_acc"].append(val_acc if val_acc is not None else 0.0)

    # Saves the model weights to a file
    def save_model(self, model):
        path = os.path.join(self.output_dir, f"{self.model_name}.pt")
        model_cpu = model.to("cpu")
        state_dict = model_cpu.state_dict()

        if self.use_half:
            state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
        torch.save(state_dict, path)
        model.to(self.device)

    # Loads the stored weights into the given model
    def load_model(self, model):
        path = os.path.join(self.output_dir, f"{self.model_name}.pt")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model.to(self.device)

    # Saves the metrics (learning curve) to a file
    def save_metrics(self):
        path = os.path.join(self.output_dir, f"{self.model_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # Loads the metrics (learning curve) from a file
    def load_metrics(self):
        path = os.path.join(self.output_dir, f"{self.model_name}_metrics.json")
        with open(path, "r") as f:
            self.history = json.load(f)
        if "test_acc" in self.history:
            print(f"Test Accuracy: {self.history['test_acc']:.4f}")

    # Plots the learning curves based on the stored metrics
    def plot_learning_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(8, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label='Train Loss')
        plt.plot(epochs, self.history["val_loss"], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label='Train Accuracy')
        plt.plot(epochs, self.history["val_acc"], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"{self.model_name}_learning_curves.png")
        plt.savefig(filename)
        plt.show()

    # Returns the best (smoothed) validation accuracy and the epoch at which it was recorded
    def get_best_validation_accuracy(logger, window_size=3):
        val_accuracies = logger.history["val_acc"]

        if len(val_accuracies) < window_size:
            # Not enough data points, return the max
            return max(val_accuracies), val_accuracies.index(max(val_accuracies))

        # Calculate moving average
        moving_averages = []
        for i in range(len(val_accuracies) - window_size + 1):
            window = val_accuracies[i:i + window_size]
            moving_averages.append(sum(window) / window_size)

        # Find best moving average
        best_avg = max(moving_averages)
        best_window_end = moving_averages.index(best_avg) + window_size - 1

        print(f"Best validation accuracy: {best_avg:.4f} at epoch {best_window_end} (smoothed over {window_size} epochs)")

    @torch.no_grad()
    def evaluate_on_testset(self, model):
        try:
            model = model.to(self.device)
            model.eval()

            total_correct, total_samples = 0, 0
            for batch in tqdm(self.test_loader, desc="Evaluating on test data", leave=False):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                preds = logits.detach().cpu().argmax(dim=1)
                y_cpu = y.detach().cpu()
                total_correct += (preds == y_cpu).sum().item()
                total_samples += y_cpu.size(0)

            accuracy = total_correct / total_samples
            self.history["test_acc"] = accuracy
            print(f"Test Accuracy: {accuracy:.4f}")
        except RuntimeError as e:
            print(f"Could not evaluate on test set. Is the model architecture correct?\n \033[91m Error: {e} \033[0m ")

    # At the end of a training run, this stores the model weights and metrics and returns a metric plot
    def finalize(self, model):
        self.evaluate_on_testset(model)
        self.save_metrics()
        self.save_model(model)
        self.plot_learning_curves()
        self.get_best_validation_accuracy()

    # Reports on the metric of a trained model
    def report(self):
        self.load_metrics()
        self.plot_learning_curves()
        self.get_best_validation_accuracy()

    # --- PyTorch Lightning integration ---

    # Callback function to store the training loss and accuracy at the end of each epoch
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return # Don't log anything during a sanity check

        # Store the train metrics at the end of training epoch
        train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
        train_acc = float(trainer.callback_metrics.get("train_acc", 0.0))
        # Save these for later use when validation completes
        self.current_train_loss = train_loss
        self.current_train_acc = train_acc

        # Use the stored training metrics
        # on_validation_epoch_end runs first, so we can use the stored values here
        self.log_epoch(train_loss, self.current_val_loss, train_acc, self.current_val_acc)

    # Callback function to store the validation loss and accuracy at the end of each epoch
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return # Don't log anything during a sanity check

        # Get validation metrics from trainer's callback_metrics
        val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
        val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))

        self.current_val_loss = val_loss
        self.current_val_acc = val_acc

    # Callback to store the model and metrics at the end of training
    def on_train_end(self, trainer, pl_module):
        self.finalize(pl_module)

# Example data loader. Feel free to write your own
class ImageDataset(Dataset):
    """ Helper class for loading images into a Torch dataset. """
    def __init__(self, X, y, transform=None):
        self.file_paths = X["file_path"].values
        self.labels = y.values
        self.transform = transform or transforms.ToTensor()

        # Few few-shot learning only
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

    # Utility function to get all images for a given class
    # Useful for few shot learning
    def get_images_for_class(self, class_label):
        return self.class_to_indices[class_label]

# Helper functions for illustration
def dummy_train(epoch):
    dummy_loss = 1.0 / (epoch + 1)
    dummy_acc = 0.5 + 0.4 * (1 - math.exp(-0.5 * epoch))
    return dummy_loss, dummy_acc
def dummy_val(epoch):
    dummy_loss = 1.2 / (epoch + 1)
    dummy_acc = 0.4 + 0.4 * (1 - math.exp(-0.5 * epoch))
    return dummy_loss, dummy_acc