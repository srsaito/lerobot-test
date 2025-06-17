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

### Strategy 2: Automated Workflow (Alternative)

**One-time setup:**
```bash
source setup_aliases.sh
```

**Then forever after, just run:**
```bash
pull-upstream-with-tensorboard
```

This single command will:
1. 🔄 Fetch upstream changes
2. 🔀 Merge upstream/main  
3. 🔧 Apply TensorBoard patch automatically
4. ✅ Commit the changes with proper message

### Strategy 3: Git Stash Method

**Workflow:**
```bash
# Before pulling upstream - save your TensorBoard changes
./manage_tensorboard_stash.sh save

# Pull upstream changes  
git fetch upstream
git merge upstream/main

# Reapply TensorBoard support
./manage_tensorboard_stash.sh apply

# Commit and push
git add lerobot/scripts/train.py
git commit -m "Reapply TensorBoard support after upstream merge"
git push origin main
```

### Strategy 4: Custom Branch (Advanced)

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
- `manage_tensorboard_stash.sh` - Script to manage TensorBoard changes via stash

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

## Automation

You can add this to your `.bashrc` or `.zshrc` for convenience:

```bash
alias pull-upstream='git fetch upstream && git merge upstream/main && ./apply_tensorboard_patch.sh'
```

Then just run `pull-upstream` whenever you want to update from upstream! 