# TensorBoard Support Workflow

This document describes how to maintain TensorBoard support in your fork while pulling upstream changes from HuggingFace lerobot.

## Problem
Upstream v0.3.3 does not ship TensorBoard logging. Your fork adds it. After upgrading, you want to preserve this functionality under the new `src/` layout.

## Solutions

### Strategy 1: Manual Git Patches (Recommended)

This approach gives you full visibility and control over each step.

**Setup (one-time):**
1. Your TensorBoard patch is saved in `tensorboard_support.patch` (updated for `src/` layout)
2. Use the script `apply_tensorboard_patch.sh` to reapply after merges

**Workflow:**
```bash
# Before pulling upstream
git status  # Make sure working directory is clean

# Pull upstream changes (or tag)
git fetch upstream --tags
git merge upstream/main  # or merge a tag (e.g., v0.3.3)

# Reapply TensorBoard support to src layout
./apply_tensorboard_patch.sh

# Commit the changes
git add src/lerobot/scripts/train.py src/lerobot/configs/default.py src/lerobot/configs/train.py src/lerobot/utils/tensorboard_utils.py
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

## Files Created / Modified by the patch

- `tensorboard_support.patch` - Patch for: `src/lerobot/scripts/train.py`, `src/lerobot/configs/{default.py,train.py}`, and adds `src/lerobot/utils/tensorboard_utils.py`.
- `combined_support.patch` - Includes TensorBoard plus small quality-of-life changes (e.g., device auto-detect in example). Optional.
- `apply_tensorboard_patch.sh` - Script to apply the patch automatically
- `QUICK_REFERENCE.md` - Step-by-step manual workflow checklist

## TensorBoard Changes Summary (src layout)

The patch adds:
1. `TensorBoardConfig` to `src/lerobot/configs/default.py` and inclusion in `TrainPipelineConfig` in `src/lerobot/configs/train.py`.
2. New utility `src/lerobot/utils/tensorboard_utils.py` providing `TensorBoardLogger`.
3. Wiring in `src/lerobot/scripts/train.py`: initialize when `cfg.tensorboard.enable`, log train/eval metrics and checkpoints, close at end.

## Usage

To enable TensorBoard in your training:

```yaml
# In your training config (YAML notation for illustration)
tensorboard:
  enable: true
  log_dir: null       # defaults to outputs/train/.../tensorboard
  comment: null
  flush_secs: 120
  disable_artifact: false
```

Then run:
```bash
tensorboard --logdir=path/to/your/output/tensorboard
```

## Quick Reference

For step-by-step instructions, see `QUICK_REFERENCE.md` which provides a detailed checklist for the manual workflow. 