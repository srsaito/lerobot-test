# TensorBoard + MPS Support Manual Workflow - Quick Reference

## 📋 Step-by-Step Checklist

When pulling upstream changes:

### 1. Check Status
```bash
git status  # Make sure working directory is clean
```

### 2. Pull Upstream Changes
```bash
git fetch upstream
git merge upstream/main
```
**What this does:** Gets latest changes from HuggingFace lerobot repository

### 3. Apply Combined Patches
```bash
./apply_tensorboard_patch.sh
```
**What this does:** 
- ✅ Adds TensorBoard import and logging support
- ✅ Adds TensorBoard logger initialization  
- ✅ Adds training metrics logging
- ✅ Adds evaluation metrics logging
- ✅ Adds proper cleanup
- ✅ Adds MPS device auto-detection for Apple Silicon
- ✅ Fixes non_blocking parameter for MPS compatibility
- ✅ Adds visualization dependencies

### 4. Review Changes (Optional)
```bash
git diff lerobot/scripts/train.py examples/2_evaluate_pretrained_policy.py
```
**What to look for:** TensorBoard code and MPS device support added in the right places

### 5. Commit Changes
```bash
git add lerobot/scripts/train.py examples/2_evaluate_pretrained_policy.py pyproject.toml
git commit -m "Reapply TensorBoard, visualization, and MPS support after upstream merge"
```

### 6. Push to Your Fork
```bash
git push origin main
```

## 🔍 What Gets Added

The patch adds these key components:

**To `train.py`:**
1. **Import:** `from lerobot.common.utils.tensorboard_utils import TensorBoardLogger`
2. **Logger setup:** Initializes TensorBoard logger based on config
3. **Training logging:** Logs metrics during training loop
4. **Evaluation logging:** Logs metrics during evaluation
5. **Cleanup:** Properly closes TensorBoard logger

**To `examples/2_evaluate_pretrained_policy.py`:**
1. **Device auto-detection:** CUDA → MPS → CPU fallback
2. **MPS compatibility:** Conditional `non_blocking` parameter usage
3. **Device logging:** Shows which device is being used

**To `pyproject.toml`:**
1. **Visualization deps:** matplotlib and seaborn for plotting

## 🚨 If Patch Fails

If `./apply_tensorboard_patch.sh` reports conflicts:

```bash
# Try 3-way merge
git apply --3way tensorboard_support.patch

# Or check what changed upstream
git diff upstream/main -- lerobot/scripts/train.py
```

## 💡 Pro Tips

- Run `git status` between steps to see what changed
- Use `git diff` to review changes before committing
- The patch is designed to be robust against upstream changes
- Keep `tensorboard_support.patch` updated if you modify TensorBoard code 