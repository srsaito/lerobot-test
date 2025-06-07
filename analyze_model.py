import os
import torch
import safetensors.torch

# Replace with your model path
model_path = "outputs/train/pi0_koch_test_6/checkpoints/last/pretrained_model/model.safetensors"

# Get file size
file_size_bytes = os.path.getsize(model_path)
file_size_mb = file_size_bytes / (1024 * 1024)
file_size_gb = file_size_mb / 1024

print(f"Model file size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")

# Load and analyze the model
state_dict = safetensors.torch.load_file(model_path)

# Count parameters
total_params = 0
for name, param in state_dict.items():
    param_size = param.numel()
    total_params += param_size
    if param_size > 1000000:  # Show large layers
        print(f"Large layer: {name} - {param_size:,} parameters - {param.dtype}")

print(f"\nTotal parameters: {total_params:,}")
print(f"Approx. FP32 memory usage: {total_params * 4 / (1024**3):.2f} GB")
print(f"Approx. FP16 memory usage: {total_params * 2 / (1024**3):.2f} GB")

        # Estimate inference memory
inference_overhead = 1.5  # Multiplier for activations, buffers, etc.
print(f"\nEstimated FP32 inference memory: {total_params * 4 * inference_overhead / (1024**3):.2f} GB")
print(f"Estimated FP16 inference memory: {total_params * 2 * inference_overhead / (1024**3):.2f} GB")
