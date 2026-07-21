"""
Fine-tune model_mobilenetv3.pt on crops extracted via extract_finetune_clips.py.

The train/val/test split was already done at the CLIP level during
extraction, so this script just loads the three folders as-is -- it does
NOT re-split, to avoid leaking frames from the same clip across splits.

Usage:
    python finetune_on_video_crops.py \
        --crops-dir data/dataset-from-video-finetune \
        --checkpoint model_mobilenetv3.pt \
        --output model_mobilenetv3_finetuned.pt
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from torch.utils.data import DataLoader

from data_processing.functions import *
from evaluation.evaluation import *
from feature_processing.functions import *
from models.training_testing import *
from models.model_framework import ImageDataset

import torch
print(torch.__version__)          # should now show something like 2.x.x+cu121
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))  # should print "NVIDIA RTX A1000 6GB Laptop GPU"

CLASS_TO_IDX = {"OTHER": 0, "PET": 1}


def load_split_df(split_dir):
    split_dir = Path(split_dir)
    data = []
    for class_name in sorted(os.listdir(split_dir)):
        class_path = split_dir / class_name
        if not class_path.is_dir():
            continue
        if class_name not in CLASS_TO_IDX:
            print(f"WARNING: folder '{class_name}' not in CLASS_TO_IDX, skipping.")
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".jpg", ".jpeg")):
                data.append({
                    "file_path": str(class_path / img_name),
                    "label": CLASS_TO_IDX[class_name],
                })
    df = pd.DataFrame(data)
    return df


def build_model(checkpoint_path, num_classes, device):
    model = mobilenet_v3_large(weights=None, num_classes=1000)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    return model


def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 on pre-split video crops")
    parser.add_argument("--crops-dir", default="data/dataset-from-video-finetune",
                         help="Folder containing train/, val/, test/ subfolders (from extract_finetune_clips.py)")
    parser.add_argument("--checkpoint", default="model_mobilenetv3.pt")
    parser.add_argument("--output", default="model_mobilenetv3_finetuned.pt")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning LR, kept low on purpose")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze conv layers, only train classifier head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    crops_dir = Path(args.crops_dir)
    if not crops_dir.exists():
        raise FileNotFoundError(
            f"'{crops_dir}' does not exist. Run extract_finetune_clips.py first, e.g.:\n"
            f"  python extract_finetune_clips.py --video-root data/fine-tuning --output-dir {crops_dir}"
        )
    for split in ["train", "val", "test"]:
        split_path = crops_dir / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"'{split_path}' does not exist -- extraction may have failed or not been run yet."
            )

    train_df = load_split_df(crops_dir / "train")
    val_df = load_split_df(crops_dir / "val")
    test_df = load_split_df(crops_dir / "test")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df) == 0:
            raise RuntimeError(f"No crops found in {crops_dir / name}. Run extract_finetune_clips.py first.")
        print(f"{name}: {len(df)} crops -- {dict(df['label'].value_counts())}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageDataset(train_df[["file_path"]], train_df["label"], transform=train_transform)
    val_ds = ImageDataset(val_df[["file_path"]], val_df["label"], transform=val_test_transform)
    test_ds = ImageDataset(test_df[["file_path"]], test_df["label"], transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    idx_to_label = {v: k for k, v in CLASS_TO_IDX.items()}

    model = build_model(args.checkpoint, args.num_classes, device)
    if args.freeze_backbone:
        freeze_backbone(model, freeze=True)
        print("Backbone frozen -- only classifier head will be fine-tuned.")

    logger = ModelLogger("model_mobilenetv3_finetuned", test_loader, device=device)

    # NOTE: check your train_model() signature -- if it hardcodes its own
    # optimizer/LR internally rather than accepting one, add an lr param
    # there, or this fine-tune will use whatever LR was set for
    # training-from-scratch (likely too high for fine-tuning).
    try:
        train_model(
            model, device, logger,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            idx_to_label=idx_to_label,
            lr=args.lr,
        )
    except TypeError:
        print("train_model() doesn't accept an lr kwarg -- falling back to its default LR.")
        print("For proper fine-tuning, check models/training_testing.py and add an lr parameter there.")
        train_model(
            model, device, logger,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            idx_to_label=idx_to_label,
        )

    torch.save(model.state_dict(), args.output)
    print(f"Fine-tuned model saved to {args.output}")

    print("\nEvaluating on held-out test clips...")
    evaluate_binary_performance(model, device, test_loader)
    plot_confusion_matrix(model, device, test_loader, idx_to_label)


if __name__ == "__main__":
    main()