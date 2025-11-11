#!/bin/bash

# Script to apply TensorBoard support patches after upstream merges
# Compatible with LeRobot v0.4.1+
# Usage: ./apply_tensorboard_patch.sh

set -e  # Exit on error

PATCH_FILE="tensorboard_v0.4.1.patch"

echo "Applying TensorBoard support patches for LeRobot v0.4.1..."

# Check if patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "❌ Error: $PATCH_FILE not found!"
    echo "Make sure you're in the root directory of the lerobot repository."
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes."
    echo "The patch will be applied on top of your current changes."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Try to apply the patch
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "✅ TensorBoard patch applied successfully!"
    echo ""
    echo "Modified files:"
    echo "  - src/lerobot/configs/default.py (TensorBoardConfig)"
    echo "  - src/lerobot/configs/train.py (tensorboard field)"
    echo "  - src/lerobot/scripts/lerobot_train.py (TensorBoard logging)"
    echo "  - src/lerobot/utils/tensorboard_utils.py (new file)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the changes: git diff"
    echo "  2. Commit the changes:"
    echo "     git add src/lerobot/configs/default.py src/lerobot/configs/train.py \\"
    echo "             src/lerobot/scripts/lerobot_train.py src/lerobot/utils/tensorboard_utils.py"
    echo "     git commit -m 'Apply TensorBoard support for v0.4.1'"
else
    echo "❌ Patch cannot be applied cleanly."
    echo ""
    echo "Possible reasons:"
    echo "  - The patch has already been applied"
    echo "  - Upstream has made conflicting changes"
    echo "  - You're not on the correct version"
    echo ""
    echo "Try one of these options:"
    echo "  1. Use 3-way merge: git apply --3way $PATCH_FILE"
    echo "  2. Check current version: git describe --tags"
    echo "  3. Review what's already applied: git diff v0.4.1"
    exit 1
fi 