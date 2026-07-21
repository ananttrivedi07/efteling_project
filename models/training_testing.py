# Create logger with model name
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
import copy
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def train_model(model, device, logger, epochs=10, train_loader=None, val_loader=None, idx_to_label=None):
    model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(idx_to_label)).to(device)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        accuracy_metric.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            total_train_loss += loss.item()
            accuracy_metric.update(outputs, y_batch)

        train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_metric.compute().item()

        model.eval()
        total_val_loss = 0
        accuracy_metric.reset()

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)
                total_val_loss += loss.item()
                accuracy_metric.update(val_outputs, y_val)

        val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_metric.compute().item()
        scheduler.step(val_loss)
        
        # Save the best weights if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"--> Best model updated at epoch {epoch} (Val Loss: {val_loss:.4f})")

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch}: Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        logger.log_epoch(train_loss, val_loss, train_acc, val_acc)

    model.load_state_dict(best_model_wts)
    logger.finalize(model)
    print("Training complete. Best weights restored.")
    
def test_model(model, device, test_loader, idx_to_label):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(idx_to_label)).to(device)

    test_loss = 0
    accuracy_metric.reset()

    with torch.no_grad():
        for X_test, y_test in tqdm(test_loader, desc="Testing"):
            X_test, y_test = X_test.to(device), y_test.to(device)

            outputs = model(X_test)
            loss = criterion(outputs, y_test)

            test_loss += loss.item()
            accuracy_metric.update(outputs, y_test)

    avg_loss = test_loss / len(test_loader)
    avg_acc = accuracy_metric.compute().item()

    print("\n" + "="*25)
    print(f"FINAL TEST RESULTS")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc:.4f}")
    print("="*25 + "\n")

    return avg_loss, avg_acc

def evaluate_binary_performance(model, device, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate the report
    report = classification_report(all_labels, all_preds, target_names=['Other', 'Paper'])
    
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\nDetailed Binary Metrics for 'PET':")
    print(f"Precision: {precision:.4f} (Percentage of correctly identified PET)")
    print(f"Recall:    {recall:.4f} (Percentage of PET collected)")
    print(f"F1-Score:  {f1:.4f} (Overall balance)")
    print("\nFull Classification Report:")
    print(report)