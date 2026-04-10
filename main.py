from data_processing.functions import *
import yaml
import os
CONFIG_file = "config.yaml"

def load_config(config_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_file)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_pipeline(config_file=None):
    config = load_config(config_file)

    # 1. data preprocessing
    if config['flags']['processing_files']:
        run_data_processing(root_folder="data", create_plots=True)
    
    # 2. feature extraction
    
    # 3. run model
    
    # 4. save results
    print("Data pipeline completed successfully!")
    

if __name__ == "__main__":
    print("Welcome to the Efteling project!")
    run_pipeline(config_file=CONFIG_file)
    
    