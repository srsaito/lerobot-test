# LeRobot Fork Enhancement Guide

This fork extends the upstream LeRobot v0.3.3 with additional features including TensorBoard support, Apple MPS (Metal Performance Shaders) compatibility, and visualization enhancements.

## 🚀 Quick Setup After Cloning

When setting up this fork on a new machine:

1. **Install dependencies:** Follow standard LeRobot installation
2. **Apply patches:** Run `./apply_tensorboard_patch.sh` to enable all enhancements
3. **Verify setup:** Check that TensorBoard utilities are available

## 🔧 Enhancements Included

### TensorBoard Support
- Complete logging integration for training and evaluation metrics
- Configurable logging parameters (log directory, flush intervals, etc.)
- Automatic cleanup and proper resource management
- Located in `src/lerobot/utils/tensorboard_utils.py`

### Apple MPS Support
- Automatic device detection: CUDA → MPS → CPU fallback
- MPS-compatible parameter handling (conditional `non_blocking`)
- Enhanced device logging for debugging

### Visualization Dependencies
- matplotlib and seaborn integration for plotting
- Additional dependencies in `pyproject.toml`

## 📋 Maintaining Your Fork

### After Pulling Upstream Changes

Follow this checklist when merging from upstream:

1. **Check status**
   ```bash
   git status  # Ensure clean working directory
   ```

2. **Pull upstream changes**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

3. **Reapply enhancements**
   ```bash
   ./apply_tensorboard_patch.sh
   ```

4. **Commit changes**
   ```bash
   git add src/lerobot/scripts/train.py examples/2_evaluate_pretrained_policy.py pyproject.toml
   git commit -m "Reapply TensorBoard, visualization, and MPS support after upstream merge"
   ```

5. **Push to fork**
   ```bash
   git push origin main
   ```

### Patch Files

- `combined_support.patch` - Main patch file containing all enhancements
- `tensorboard_support.patch` - TensorBoard-only patch (if needed separately)
- `visualization_support.patch` - Visualization dependencies only
- `apply_tensorboard_patch.sh` - Automated application script

## 📊 Using TensorBoard

### Configuration
Enable TensorBoard in your training config:

```yaml
tensorboard:
  enable: true
  log_dir: null       # defaults to outputs/train/.../tensorboard
  comment: null
  flush_secs: 120
  disable_artifact: false
```

### Usage
```bash
# Start TensorBoard server
tensorboard --logdir=path/to/your/output/tensorboard

# Or use the default output location
tensorboard --logdir=outputs/train/
```

### What Gets Logged
- Training loss and metrics
- Evaluation results
- Learning rate schedules
- Model checkpoints and parameters

## 🔍 Technical Details

### Files Modified by Patches

**Training Script (`src/lerobot/scripts/train.py`):**
- TensorBoard logger initialization
- Training metrics logging
- Evaluation metrics logging
- Proper cleanup and resource management

**Evaluation Example (`examples/2_evaluate_pretrained_policy.py`):**
- MPS device auto-detection
- MPS-compatible tensor operations
- Enhanced device logging

**Configuration (`pyproject.toml`):**
- Visualization dependencies (matplotlib, seaborn)

**New Files Added:**
- `src/lerobot/utils/tensorboard_utils.py` - TensorBoard logging utilities

### Device Compatibility
The fork automatically detects and uses the best available device:
1. NVIDIA CUDA (if available)
2. Apple MPS (on Apple Silicon Macs)
3. CPU (fallback)

## 🚨 Troubleshooting

### Patch Application Fails
If `./apply_tensorboard_patch.sh` reports conflicts:

```bash
# Try 3-way merge
git apply --3way combined_support.patch

# Check what changed upstream
git diff upstream/main -- src/lerobot/scripts/train.py

# Manual resolution may be needed for significant upstream changes
```

### MPS Issues
If you encounter MPS-related errors:
- Ensure you're on macOS with Apple Silicon
- Check PyTorch MPS support: `torch.backends.mps.is_available()`
- Fall back to CPU if needed by setting `PYTORCH_ENABLE_MPS_FALLBACK=1`

## 📝 Notes

- This fork tracks upstream `v0.3.3` with the new `src/` layout
- Patches are designed to be robust against upstream changes
- Keep patch files updated if you modify the enhanced features
- All enhancements are backward-compatible with standard LeRobot usage