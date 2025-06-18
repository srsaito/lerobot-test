#!/bin/bash

# Script to apply TensorBoard, visualization, and MPS support patches after upstream merges
# Usage: ./apply_tensorboard_patch.sh

echo "Applying TensorBoard, visualization, and MPS support patches..."

# Check if patch file exists
if [ ! -f "combined_support.patch" ]; then
    echo "Error: combined_support.patch not found!"
    exit 1
fi

# Apply the patch
git apply combined_support.patch

if [ $? -eq 0 ]; then
    echo "✅ Combined patch applied successfully!"
    echo "You can now commit the changes with:"
    echo "  git add lerobot/scripts/train.py examples/2_evaluate_pretrained_policy.py pyproject.toml"
    echo "  git commit -m 'Apply TensorBoard, visualization, and MPS support patches'"
else
    echo "❌ Patch failed to apply. You may need to resolve conflicts manually."
    echo "Try: git apply --3way combined_support.patch"
fi 