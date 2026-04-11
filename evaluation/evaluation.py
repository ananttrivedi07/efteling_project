import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def plot_confusion_matrix(model, device, loader, idx_to_label):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create the matrix
    cm = confusion_matrix(all_labels, all_preds)
    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: TrashNet 6-Class')
    # plt.show()
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved.")