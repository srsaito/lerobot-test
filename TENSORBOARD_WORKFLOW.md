# TensorBoard Support Workflow

This document describes how to maintain TensorBoard support in your fork while pulling upstream changes from HuggingFace lerobot.

## Problem
Every time you pull upstream changes, your TensorBoard modifications to `lerobot/scripts/train.py` get overwritten because upstream doesn't have TensorBoard support.

## Solutions

### Strategy 1: Manual Git Patches (Recommended)

This approach gives you full visibility and control over each step.

**Setup (one-time):**
1. Your TensorBoard patch is saved in `tensorboard_support.patch`
2. Use the script `apply_tensorboard_patch.sh` to reapply after merges

**Workflow:**
```bash
# Before pulling upstream
git status  # Make sure working directory is clean

# Pull upstream changes
git fetch upstream
git merge upstream/main

# Reapply TensorBoard support
./apply_tensorboard_patch.sh

# Commit the changes
git add lerobot/scripts/train.py
git commit -m "Reapply TensorBoard support after upstream merge"

# Push to your fork
git push origin main
```

### Alternative: Custom Branch (Advanced)

Create a dedicated branch for TensorBoard support:

```bash
# Create a feature branch for TensorBoard
git checkout -b feature/tensorboard-support
git add lerobot/scripts/train.py
git commit -m "Add TensorBoard support"

# When pulling upstream
git checkout main
git pull upstream main
git checkout feature/tensorboard-support
git rebase main

# Merge back to main
git checkout main
git merge feature/tensorboard-support
```

## Files Created

- `tensorboard_support.patch` - Contains the TensorBoard changes as a patch
- `apply_tensorboard_patch.sh` - Script to apply the patch automatically
- `QUICK_REFERENCE.md` - Step-by-step manual workflow checklist

## TensorBoard Changes Summary

The patch adds:
1. Import: `from lerobot.common.utils.tensorboard_utils import TensorBoardLogger`
2. Logger initialization based on `cfg.tensorboard.enable`
3. Training metrics logging to TensorBoard
4. Evaluation metrics logging to TensorBoard  
5. Proper cleanup of TensorBoard logger

## Usage

To enable TensorBoard in your training:

```yaml
# In your training config
tensorboard:
  enable: true
  log_dir: null  # Optional: defaults to output_dir/tensorboard
```

Then run:
```bash
tensorboard --logdir=path/to/your/output/tensorboard
```

## Quick Reference

For step-by-step instructions, see `QUICK_REFERENCE.md` which provides a detailed checklist for the manual workflow. 