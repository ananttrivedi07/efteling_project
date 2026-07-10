from data_processing.functions import *
from evaluation.evaluation import *
from feature_processing.functions import *
from models.training_testing import *
import yaml
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

CONFIG_file = "config.yaml"
NUM_CLASSES = 2

def load_config(config_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_file)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_model(model_path, device, num_classes):
    model = mobilenet_v3_large(num_classes=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model

def run_pipeline(config_file=None):
    config = load_config(config_file)

    # 1. data preprocessing
    if config['flags']['processing_files']:
        run_data_processing(root_folder="data", show_raw_data=True)
    
    # 2. feature extraction
    if config['flags']['generate_features']:
        train_loader, val_loader, test_loader, idx_to_label = run_feature_processing(root_folder="data", show_training_data=config['flags']['show_training_data'], show_raw_data=config['flags']['show_raw_data'])
    
    # 3. run model
    if config['flags']['train_model']:
        logger = ModelLogger("model_mobilenetv3", test_loader)
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        num_classes, num_epochs, batch_size = NUM_CLASSES, 30, 32
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # Adjust the final layer for our classes
        model.to(device)
        train_model(model, device, logger, epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, idx_to_label=idx_to_label)
        print("Model training completed!")
    
    if config['flags']['only_test_model']:
        print("Testing the model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = NUM_CLASSES
        model = load_model("model_mobilenetv3.pt", device, num_classes)
        test_model(model, device, test_loader, idx_to_label)
        
        # analyzing individal classes.
        evaluate_binary_performance(model, device, test_loader)
        plot_confusion_matrix(model, device, test_loader, idx_to_label)
    # 4. save results
    print("Data pipeline completed successfully!")
    

if __name__ == "__main__":
    print("Welcome to the Efteling project!")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    run_pipeline(config_file=CONFIG_file)