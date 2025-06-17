#!/bin/bash

# Script to apply TensorBoard support patch after upstream merges
# Usage: ./apply_tensorboard_patch.sh

echo "Applying TensorBoard support patch..."

# Check if patch file exists
if [ ! -f "tensorboard_support.patch" ]; then
    echo "Error: tensorboard_support.patch not found!"
    exit 1
fi

# Apply the patch
git apply tensorboard_support.patch

if [ $? -eq 0 ]; then
    echo "✅ TensorBoard patch applied successfully!"
    echo "You can now commit the changes with:"
    echo "  git add lerobot/scripts/train.py"
    echo "  git commit -m 'Apply TensorBoard support patch'"
else
    echo "❌ Patch failed to apply. You may need to resolve conflicts manually."
    echo "Try: git apply --3way tensorboard_support.patch"
fi 