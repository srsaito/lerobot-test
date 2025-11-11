# TensorBoard Support Patches for LeRobot

This directory contains patches to add TensorBoard logging support to LeRobot v0.4.1+.

## Files

- **tensorboard_v0.4.1.patch** - Clean patch file for LeRobot v0.4.1
- **apply_tensorboard_patch.sh** - Automated script to apply the patch

## What the Patch Adds

### 1. TensorBoard Configuration (`src/lerobot/configs/default.py`)
Adds `TensorBoardConfig` dataclass with options:
- `enable`: Enable/disable TensorBoard logging
- `log_dir`: Custom log directory (defaults to `outputs/train/.../tensorboard`)
- `comment`: Optional comment for the run
- `flush_secs`: How often to flush logs to disk (default: 120s)
- `disable_artifact`: Disable artifact logging

### 2. Training Config Integration (`src/lerobot/configs/train.py`)
Adds `tensorboard` field to `TrainPipelineConfig`

### 3. Training Script Integration (`src/lerobot/scripts/lerobot_train.py`)
- Initializes TensorBoard logger
- Logs training metrics per step
- Logs evaluation metrics
- Logs model checkpoints
- Proper cleanup on training completion

### 4. TensorBoard Utilities (`src/lerobot/utils/tensorboard_utils.py`)
Complete TensorBoard logging wrapper with:
- Scalar metrics logging
- Text logging
- Video logging (experimental)
- Policy checkpoint logging
- Automatic directory management

## Apple MPS Support

**Good news!** LeRobot v0.4.1+ has built-in Apple MPS (Metal Performance Shaders) support. You don't need any patches for MPS!

### What's Built-In:

1. **Auto Device Detection** (`src/lerobot/utils/utils.py`)
   - Automatically detects and uses MPS on Apple Silicon Macs
   - Fallback chain: CUDA → MPS → Intel XPU → CPU

2. **Smart Tensor Operations** (`src/lerobot/processor/device_processor.py`)
   - Only uses `non_blocking=True` for CUDA (prevents MPS errors)
   - Automatic float64 → float32 conversion for MPS compatibility

3. **Multi-Backend Support**
   - NVIDIA CUDA (if available)
   - Apple MPS (macOS with Apple Silicon)
   - Intel XPU (for Intel GPUs)
   - CPU (fallback)

### Using MPS:

Simply set device to "mps" in your config or let it auto-detect:

```bash
# Auto-detect (will use MPS on Apple Silicon)
python -m lerobot.scripts.lerobot_train policy=... env=...

# Or explicitly specify MPS
python -m lerobot.scripts.lerobot_train policy=... env=... device=mps
```

**Note:** Some policies like VQBeT don't support MPS yet. Check the error messages if you encounter issues.

## Usage

### After Pulling Upstream Updates

When you merge new changes from upstream LeRobot and TensorBoard support gets overwritten:

```bash
# 1. Merge upstream changes
git fetch upstream
git merge upstream/main  # or specific version tag

# 2. Reapply TensorBoard patch
./apply_tensorboard_patch.sh

# 3. Commit the patched changes
git add src/lerobot/configs/default.py src/lerobot/configs/train.py \
        src/lerobot/scripts/lerobot_train.py src/lerobot/utils/tensorboard_utils.py
git commit -m "Reapply TensorBoard support after upstream merge"
```

### First Time Setup

If setting up on a fresh clone of upstream LeRobot:

```bash
# 1. Clone upstream repo
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 2. Checkout v0.4.1 (or later)
git checkout v0.4.1

# 3. Copy patch files to repo root
cp /path/to/tensorboard_v0.4.1.patch .
cp /path/to/apply_tensorboard_patch.sh .

# 4. Apply patch
chmod +x apply_tensorboard_patch.sh
./apply_tensorboard_patch.sh

# 5. Commit changes
git add src/lerobot/
git commit -m "Add TensorBoard support to v0.4.1"
```

## Enabling TensorBoard in Training

Add to your training config YAML:

```yaml
tensorboard:
  enable: true
  log_dir: null       # defaults to outputs/train/.../tensorboard
  comment: "my_experiment"
  flush_secs: 120
  disable_artifact: false
```

Or via command line:

```bash
python -m lerobot.scripts.lerobot_train \
  policy=... \
  env=... \
  tensorboard.enable=true \
  tensorboard.comment="my_experiment"
```

## Viewing TensorBoard Logs

```bash
# Start TensorBoard server
tensorboard --logdir=outputs/train/

# Or specific run
tensorboard --logdir=outputs/train/your_policy_name/your_run_timestamp/tensorboard
```

Then open http://localhost:6006 in your browser.

## Maintaining the Patch

If you make improvements to the TensorBoard support, regenerate the patch:

```bash
# Make your changes to the TensorBoard files
# Then regenerate the patch from v0.4.1
git diff v0.4.1 -- \
  src/lerobot/configs/default.py \
  src/lerobot/configs/train.py \
  src/lerobot/scripts/lerobot_train.py \
  src/lerobot/utils/tensorboard_utils.py \
  > tensorboard_v0.4.1.patch
```

## Troubleshooting

### Patch fails to apply

**Check 1**: Are you on the right version?
```bash
git describe --tags  # Should show v0.4.1 or later
```

**Check 2**: Is the patch already applied?
```bash
git diff v0.4.1 src/lerobot/  # Shows current differences
```

**Check 3**: Try 3-way merge
```bash
git apply --3way tensorboard_v0.4.1.patch
# Resolve conflicts if any, then:
git add .
git am --continue
```

### TensorBoard import fails

Make sure TensorBoard is installed:
```bash
pip install tensorboard
# or
conda install tensorboard
```

### Logs not appearing

Check that TensorBoard is enabled in your config:
```bash
# In your training script output, look for:
# "TensorBoard logging enabled at: outputs/train/.../tensorboard"
```

## Version Compatibility

- **v0.4.1+**: Fully tested and working ✅ (includes built-in MPS support)
- **v0.4.0**: Not supported (use upgrade to v0.4.1+)
- **v0.3.x**: Not compatible (different file structure)

## Future Updates

When LeRobot v0.5.0 or later is released:

1. Check if TensorBoard support was added upstream
2. If not, create a new patch: `tensorboard_v0.5.0.patch`
3. Update this README with compatibility info
4. Test thoroughly before using in production
