"""
Convert a trained PyTorch MobileNetV3-Large checkpoint (.pt) to a single
self-contained ONNX file, and optionally on to TensorFlow Lite (.tflite).

Usage:
    python convert_model.py --pt model_mobilenetv3.pt --onnx model_mobilenetv3.onnx
    python convert_model.py --pt model_mobilenetv3.pt --onnx model_mobilenetv3.onnx --tflite tflite_out

Requirements:
    pip install torch torchvision onnx
    # only needed if you also want --tflite:
    pip install onnx2tf tensorflow onnx-graphsurgeon sng4onnx
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


def convert_pt_to_onnx(pt_path, onnx_path, num_classes, input_size=224, device="cpu"):
    print(f"Loading PyTorch checkpoint from {pt_path} ...")

    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    print(f"Exporting to ONNX at {onnx_path} ...")
    # dynamo=False forces the legacy TorchScript-based exporter, which embeds
    # all weights directly in the .onnx file instead of writing a separate
    # "<name>.onnx.data" external-data file (the dynamo exporter's default
    # behavior in newer torch versions).
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Sanity check + guarantee a single-file output even if some external
    # data file slipped through (e.g. from a different torch/onnx version).
    try:
        import onnx
        from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data

        onnx_model = onnx.load(onnx_path, load_external_data=True)

        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            print(f"Found external data file {data_path}, merging into single .onnx file ...")
            onnx.save_model(
                onnx_model,
                onnx_path,
                save_as_external_data=False,
            )
            os.remove(data_path)
            print("Merged and removed external data file.")

        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except ImportError:
        print("Note: 'onnx' package not installed, skipping model check / merge step.")

    print(f"ONNX export complete: {onnx_path} (single file, weights embedded).")


def convert_onnx_to_tflite(onnx_path, tflite_output_dir):
    """
    Uses the onnx2tf CLI tool to convert ONNX -> TensorFlow SavedModel -> TFLite.
    Requires: pip install onnx2tf tensorflow onnx-graphsurgeon sng4onnx
    """
    print(f"Converting {onnx_path} to TFLite via onnx2tf ...")
    os.makedirs(tflite_output_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                sys.executable, "-m", "onnx2tf",
                "-i", onnx_path,
                "-o", tflite_output_dir,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("Error: onnx2tf is not installed. Run: pip install onnx2tf tensorflow onnx-graphsurgeon sng4onnx")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"onnx2tf conversion failed: {e}")
        sys.exit(1)

    print(f"TFLite conversion complete. Output files are in: {tflite_output_dir}")
    print("Look for a float32.tflite (and possibly quantized variants) in that folder.")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX / TFLite")
    parser.add_argument("--pt", required=True, help="Path to input .pt checkpoint")
    parser.add_argument("--onnx", required=True, help="Path to write .onnx file")
    parser.add_argument("--tflite", default=None, help="Output directory for TFLite conversion (omit to skip)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size (square)")
    args = parser.parse_args()

    convert_pt_to_onnx(
        pt_path=args.pt,
        onnx_path=args.onnx,
        num_classes=args.num_classes,
        input_size=args.input_size,
    )

    if args.tflite:
        convert_onnx_to_tflite(args.onnx, args.tflite)


if __name__ == "__main__":
    main()