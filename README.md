# Efteling Honors Project: Computer Vision Pipeline

This repository contains the image classification and feature processing pipeline for the Efteling project. The model achieves an accuracy of **96.64%** and is optimized for NVIDIA GPU acceleration (CUDA 12.1).

##  Quick Start (Environment Setup)

To ensure the GPU acceleration (RTX A1000) works and to avoid NumPy 2.0 compatibility issues with `openml-pytorch`, follow these exact steps:

### 1. Create the Environment
Open your terminal (Anaconda Prompt recommended) and run:
```bash
conda create -n efteling_project python=3.10 -y
conda activate efteling_project
pip install -r requirements.txt
````

### 2. Run the pipeline
Before running the pipeline one need to set the flags in the ```config.yaml```.
If running the pipeline for the first time, one needs to set the ```processing_files``` to ``True`` in order to extract the training data from KaggleHub. 

Then you can get the other flags to generate features, training model or only testing the saved model.

```bash 
python main.py
```
