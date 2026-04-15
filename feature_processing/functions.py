import random
import pandas as pd
import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from models.model_framework import *
from torchvision import transforms

def load_image_dataset(data_dir):
    data = []

    data_dir = Path(data_dir)

    # Loop over class folders
    for class_name in os.listdir(data_dir):
        class_path = data_dir / class_name

        if not class_path.is_dir():
            continue

        # Loop over images inside class
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg')):
                file_path = class_path / img_name

                data.append({
                    "file_path": str(file_path),
                    "label": class_name
                })

    # Create DataFrame
    df = pd.DataFrame(data)

    print(f"The dataset has {len(df)} images and {df['label'].nunique()} classes")

    return df

def visualize_random(dataset, idx_to_label):
    fig, axes = plt.subplots(1, 10, figsize=(15, 5), layout='constrained')
    
    for i, n in enumerate(random.sample(range(0, len(dataset)), 10)):
        image, label = dataset[n]
        
        np_img = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Denormalize
        axes[i].imshow(np_img)
        
        axes[i].set_xlabel(idx_to_label[label.item()].split("(")[0][:20], fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.show()



def run_feature_processing(root_folder="data", show_training_data=False, show_raw_data=False):
    print("Running Binary Feature Processing (Target: PAPER)...")
    base_path = os.path.join(root_folder, "dataset-resized")
    df = load_image_dataset(base_path)

    # --- BINARY MAPPING LOGIC ---
    # Target 1 = Paper
    # Target 0 = Everything Else (Cardboard, Glass, Metal, Plastic, Trash)
    df['binary_target'] = df['label'].apply(lambda x: 1 if x == 'paper' else 0)

    X_all = df[["file_path"]]
    y_all = df["binary_target"] 

    # Simplify mappings for the 2-class setup
    label_to_idx = {"Other": 0, "Paper": 1}
    idx_to_label = {0: "Other", 1: "Paper"}
    
    print("\nNew Binary Class Distribution:")
    print(df["binary_target"].value_counts().rename(index=idx_to_label))
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = ImageDataset(X_train, y_train, transform=train_transform)
    val_ds = ImageDataset(X_val, y_val, transform=val_test_transform)
    test_ds = ImageDataset(X_test, y_test, transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    if show_training_data:
        visualize_random(train_ds, idx_to_label)
    
    return train_loader, val_loader, test_loader, idx_to_label