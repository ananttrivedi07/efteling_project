# Efteling Honors Project: Computer Vision Pipeline

This repository contains an end-to-end computer vision pipeline for binary waste-item classification. It covers dataset acquisition, feature/dataset preparation, model training (ResNet-18 and MobileNetV3-Large), fine-tuning on real deployment footage, model export for edge inference, and a live webcam demo. The current best model reaches **96.64% accuracy**, and the pipeline is set up for NVIDIA GPU acceleration (tested with CUDA 12.1).

## Overview

The pipeline follows four stages, mirrored in the repo structure below:

1. **Data processing** — download the [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet) dataset via `kagglehub` and organize it locally.
2. **Feature processing** — turn the raw multi-class dataset into a binary classification problem (currently *PET* vs. *Other*), split it into train/val/test, and wrap it in `DataLoader`s.
3. **Training & evaluation** — train a classifier (ResNet-18 or MobileNetV3-Large), track metrics, save the best checkpoint, and report precision/recall/F1 plus a confusion matrix.
4. **Fine-tuning & deployment** — fine-tune the MobileNetV3 model on crops extracted from real video footage, export it to ONNX/TFLite, and run it live from a webcam.



## Demo
 
[![Watch the demo](https://img.youtube.com/vi/-r0RdznWs1s/hqdefault.jpg)](https://www.youtube.com/shorts/-r0RdznWs1s)


## Repository Structure

```
.
├── main.py                     # Full pipeline entry point (ResNet-18 backbone)
├── main_mobilenet.py           # Full pipeline entry point (MobileNetV3-Large backbone)
├── config.yaml                 # Pipeline stage flags (see Configuration)
├── requirements.txt
├── finetune_on_video_crops.py  # Fine-tunes a trained MobileNetV3 checkpoint on real video crops
├── convert_model.py            # Exports a .pt checkpoint to ONNX (and optionally TFLite)
├── live_inference.py           # Real-time webcam demo (OpenCV)
├── data_processing/
│   └── functions.py            # Downloads TrashNet, visualizes raw samples
├── feature_processing/
│   └── functions.py            # Dataset loading, binary-label mapping, splitting, DataLoaders
├── models/
│   ├── model_framework.py      # ModelLogger (checkpointing/metrics/PL callback) + ImageDataset
│   └── training_testing.py     # train_model, test_model, evaluate_binary_performance
├── evaluation/
│   └── evaluation.py           # plot_confusion_matrix and other evaluation plots
└── data/                       # Created at runtime; holds the downloaded/organized dataset
```

> Script names above (`main_mobilenet.py`, `live_inference.py`) are inferred from the code — rename the references below if your actual filenames differ.

## Requirements

- Python 3.10
- Conda (Anaconda/Miniconda)
- NVIDIA GPU + CUDA 12.1 recommended (tested on an RTX A1000); CPU also works, just slower
- A [Kaggle account with an API token](https://www.kaggle.com/docs/api) configured, since `data_processing/functions.py` downloads TrashNet via `kagglehub` on your behalf

## Quick Start (Environment Setup)

To ensure GPU acceleration works and to avoid NumPy 2.0 compatibility issues with `openml-pytorch`, follow these exact steps:

### 1. Create the environment

```bash
conda create -n efteling_project python=3.10 -y
conda activate efteling_project
pip install -r requirements.txt
```

### 2. Configure the pipeline

Set the flags in `config.yaml` before running. On your first run, set `processing_files: true` so the raw data gets downloaded from KaggleHub. From there, use the remaining flags to control feature generation, training, and testing:

```yaml
flags:
  processing_files: true      # Download + organize the raw TrashNet dataset
  show_raw_data: false        # Plot a few raw images per class while processing
  generate_features: true     # Build the binary train/val/test DataLoaders
  show_training_data: false   # Visualize a batch of augmented training images
  train_model: true           # Train a fresh model and save the best checkpoint
  only_test_model: false      # Skip training, just evaluate a saved checkpoint
```

> `train_model` and `only_test_model` both rely on the `DataLoader`s built in the `generate_features` step, so keep `generate_features: true` whenever either of those is enabled.

### 3. Run the pipeline

```bash
python rest_net_main.py              # ResNet-18 backbone
python mobile_net_main.py    # MobileNetV3-Large backbone
```

Both scripts follow the same flow — download data → build features → train → evaluate (precision/recall/F1 + confusion matrix) — driven entirely by `config.yaml`.

## Fine-Tuning on Real Video Footage

Once you have a MobileNetV3 checkpoint trained on TrashNet, `finetune_on_video_crops.py` adapts it to crops pulled from real deployment video (extracted separately, e.g. via an `extract_finetune_clips.py` step). The train/val/test split happens at the *clip* level during extraction, so this script loads the three folders as-is rather than re-splitting — that avoids leaking frames from the same clip across splits.

```bash
python finetune_on_video_crops.py \
    --crops-dir data/dataset-from-video-finetune \
    --checkpoint model_mobilenetv3.pt \
    --output model_mobilenetv3_finetuned.pt
```

Useful flags:
- `--freeze-backbone` — only train the classifier head, keep the convolutional backbone frozen
- `--lr` — fine-tuning learning rate (defaults to a conservative `1e-4`)

This stage currently targets a `PET` vs. `OTHER` split (see `CLASS_TO_IDX` in the script) — worth double-checking against whichever binary target your TrashNet-trained checkpoint used (see the note at the bottom of this README).

## Exporting for Deployment (ONNX / TFLite)

`convert_model.py` converts a trained MobileNetV3-Large checkpoint into a single self-contained ONNX file, and optionally on to TensorFlow Lite for edge/mobile deployment:

```bash
# ONNX only
python convert_model.py --pt model_mobilenetv3_finetuned.pt --onnx model_mobilenetv3.onnx

# ONNX + TFLite
python convert_model.py --pt model_mobilenetv3_finetuned.pt --onnx model_mobilenetv3.onnx --tflite tflite_out
```

Requires `torch`, `torchvision`, and `onnx`; the `--tflite` path additionally needs `onnx2tf`, `tensorflow`, `onnx-graphsurgeon`, and `sng4onnx`.

## Live Webcam Demo

`live_inference.py` runs a trained checkpoint against a live webcam feed with OpenCV, overlaying the predicted label and confidence on each frame:

```bash
python live_inference.py
```

Press `q` to quit. By default it loads `model_resnet18.pt` and uses the DirectShow backend for the webcam (Windows) — swap in a different checkpoint or capture backend if that doesn't match your setup.

## Model Details

| | ResNet-18 | MobileNetV3-Large |
|---|---|---|
| Pretrained weights | ImageNet (`ResNet18_Weights.DEFAULT`) | ImageNet (`MobileNet_V3_Large_Weights.DEFAULT`) |
| Head | `fc` replaced with `Linear(in_features, num_classes)` | `classifier[-1]` replaced with `Linear(in_features, num_classes)` |
| Best suited for | Baseline / accuracy comparisons | Edge / mobile deployment (small, ONNX/TFLite-friendly) |

**Training setup** (`models/training_testing.py`): SGD (`lr=0.001`, `momentum=0.9`, `weight_decay=1e-4`), `ReduceLROnPlateau` scheduler on validation loss, cross-entropy loss, best-checkpoint selection by lowest validation loss, accuracy tracked via `torchmetrics`.

**Logging** (`models/model_framework.py`): `ModelLogger` saves the best model weights, a `_metrics.json` history file, and a `_learning_curves.png` plot for every run, and doubles as a PyTorch Lightning `Callback` if you train with a `Trainer` instead of the manual loop in `train_model`.

## Outputs

After a training run, for each model name you should have:
- `{model_name}.pt` — best model weights (lowest validation loss)
- `{model_name}_metrics.json` — per-epoch train/val loss and accuracy, plus final test accuracy
- `{model_name}_learning_curves.png` — loss and accuracy curves
- A confusion matrix plot from `evaluation.plot_confusion_matrix`
- Precision/recall/F1 printed via `evaluate_binary_performance`

## Note
- **kagglehub auth**: `run_data_processing` calls `kagglehub.dataset_download`, which needs a valid Kaggle API token configured on the machine running it.