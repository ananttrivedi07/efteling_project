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
    print("Running feature processing...")
    base_path = os.path.join(root_folder, "dataset-resized")
    df = load_image_dataset(base_path)

    # Equivalent to X_all and y_all
    X_all = df[["file_path"]]
    y_all = df["label"]

    # Map labels to integers
    label_to_idx = {label: idx for idx, label in enumerate(sorted(y_all.unique()))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_all = y_all.map(label_to_idx)
    print("\nClass distribution (original labels):")
    print(df["label"].value_counts())
    # Train-test split and data loaders
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5]*3, [0.5]*3)
    # ])
    
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224), # EXPERIMENT02
    #     # transforms.Resize((224,224)),
    #     # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    #     # transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3,sigma=(0.2, 5))]),p=0.15),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     #transforms.RandomCrop(128),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    # ])

    # FOR TRAINING ONLY
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # FOR VALIDATION AND TESTING
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Just resize it to the expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    train_ds = ImageDataset(X_train, y_train, transform=train_transform)
    val_ds = ImageDataset(X_val, y_val, transform=val_test_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Test data loader. Leave this unchanged.
    test_ds = ImageDataset(X_test, y_test, transform=val_test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    
    if show_training_data:
        visualize_random(train_ds, idx_to_label)
    
    print("Classes:", label_to_idx)
    return train_loader, val_loader, test_loader, idx_to_label