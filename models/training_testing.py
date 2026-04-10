# Create logger with model name
import torch
import torch.optim as optim
import torchmetrics
from torchvision import models
from models.model_framework import *
from models.RestNet18 import ResNet18
from torch import nn

# Define how to train your model
def train_model(model, device, logger, epochs=10, train_loader=None, val_loader=None, idx_to_label=None):
    #Train the model
    train_loss = []
    val_loss = []
    train_acc_list = []
    train_lost_list = []
    val_acc_list = []
    val_lost_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.0001, momentum = 0.9)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(idx_to_label)).to(device)
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass + loss calculation
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training metrics
            total_loss += loss.item()
            accuracy_metric.reset()
            correct += accuracy_metric(outputs, y_batch).item() * y_batch.size(0)

            total += y_batch.size(0)

        # Compute epoch training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_acc_list.append(train_acc)
        train_lost_list.append(train_loss)
        print(f'This is the training loss: {train_loss}, for epoch: {epoch}')
        print(f'This is train accuracy: {train_acc}, for epoch: {epoch}')

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)

                val_loss += loss.item()
                accuracy_metric.reset()
                val_correct += accuracy_metric(val_outputs, y_val).item() * y_val.size(0)

                val_total += y_val.size(0)

        val_avg_loss = val_loss / len(val_loader)
        val_avg_acc = val_correct / val_total
        val_acc_list.append(val_avg_acc)
        val_lost_list.append(val_avg_loss)
        print(f'This is validation loss: {val_avg_loss}, for epoch: {epoch}')
        print(f'This is validation accuracy: {val_avg_acc}, for epoch: {epoch}')
        # Log
        logger.log_epoch(train_loss, val_avg_loss, train_acc, val_avg_acc)

    # Save
    logger.finalize(model)
    
    
def test_model(model, device, test_loader, idx_to_label):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=len(idx_to_label)
    ).to(device)

    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            outputs = model(X_test)
            loss = criterion(outputs, y_test)

            test_loss += loss.item()
            accuracy_metric.reset()
            test_correct += accuracy_metric(outputs, y_test).item() * y_test.size(0)
            test_total += y_test.size(0)

    avg_loss = test_loss / len(test_loader)
    avg_acc = test_correct / test_total

    print("\n===== TEST RESULTS =====")
    print(f"Test Loss: {avg_loss}")
    print(f"Test Accuracy: {avg_acc}")

    return avg_loss, avg_acc
