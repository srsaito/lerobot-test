# LeRobot Fork Enhancement Guide

This fork extends the upstream LeRobot v0.4.1+ with TensorBoard support. Apple MPS (Metal Performance Shaders) support is now built into upstream LeRobot v0.4.1+.

## 🚀 Quick Setup After Cloning

When setting up this fork on a new machine:

### Environment Setup
1. **Check for local environment config:** Look for `CLAUDE.local.md` to see which conda environment to activate
   - If `CLAUDE.local.md` exists: Follow the conda environment specified there
   - If no `CLAUDE.local.md`: Ask the user which conda environment to use for this project

2. **First-time installation:** When installing LeRobot for the first time:
   ```bash
   # Create a dedicated conda environment
   conda create -n lerobot python=3.9
   conda activate lerobot
   
   # Install in editable mode with extras
   pip install -e .[pusht]
   ```
   
   Then create `CLAUDE.local.md` with:
   ```markdown
   # Local Development Setup
   
   Always activate the lerobot conda environment before working on this project:
   ```bash
   conda activate lerobot
   ```
   ```

### Installation Steps
3. **Install dependencies:** Follow standard LeRobot installation or use environment from step 1-2
4. **Apply patches:** Run `./apply_tensorboard_patch.sh` to enable all enhancements
5. **Verify setup:** Check that TensorBoard utilities are available

## 🔧 Enhancements Included

### TensorBoard Support
- Complete logging integration for training and evaluation metrics
- Configurable logging parameters (log directory, flush intervals, etc.)
- Automatic cleanup and proper resource management
- Located in `src/lerobot/utils/tensorboard_utils.py`

### Apple MPS Support (Built-in as of v0.4.1)
LeRobot v0.4.1+ includes native MPS support - **no patches needed!**
- Automatic device detection: CUDA → MPS → Intel XPU → CPU fallback
- MPS-compatible parameter handling (conditional `non_blocking`)
- Automatic float64 → float32 conversion for MPS
- Enhanced device logging for debugging

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
   git add src/lerobot/configs/default.py src/lerobot/configs/train.py \
           src/lerobot/scripts/lerobot_train.py src/lerobot/utils/tensorboard_utils.py
   git commit -m "Reapply TensorBoard support after upstream merge"
   ```

5. **Push to fork**
   ```bash
   git push origin main
   ```

### Patch Files

- `tensorboard_v0.4.1.patch` - Clean patch file for LeRobot v0.4.1+
- `apply_tensorboard_patch.sh` - Automated application script
- `PATCHES_README.md` - Detailed documentation for the patch system

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

**Configuration Files:**
- `src/lerobot/configs/default.py` - Adds `TensorBoardConfig` dataclass
- `src/lerobot/configs/train.py` - Adds `tensorboard` field to `TrainPipelineConfig`

**Training Script (`src/lerobot/scripts/lerobot_train.py`):**
- TensorBoard logger initialization
- Training metrics logging
- Evaluation metrics logging
- Proper cleanup and resource management

**New Files Added:**
- `src/lerobot/utils/tensorboard_utils.py` - TensorBoard logging utilities

### Device Compatibility (Built-in as of v0.4.1)
LeRobot v0.4.1+ automatically detects and uses the best available device:
1. NVIDIA CUDA (if available)
2. Apple MPS (on Apple Silicon Macs)
3. Intel XPU (for Intel GPUs)
4. CPU (fallback)

## 🚨 Troubleshooting

### Patch Application Fails
If `./apply_tensorboard_patch.sh` reports conflicts:

```bash
# Try 3-way merge
git apply --3way tensorboard_v0.4.1.patch

# Check what changed upstream
git diff upstream/main -- src/lerobot/scripts/lerobot_train.py

# Manual resolution may be needed for significant upstream changes
# See PATCHES_README.md for more troubleshooting tips
```

### MPS Issues
If you encounter MPS-related errors:
- Ensure you're on macOS with Apple Silicon
- Check PyTorch MPS support: `torch.backends.mps.is_available()`
- Fall back to CPU if needed by setting `PYTORCH_ENABLE_MPS_FALLBACK=1`

## 🤖 Claude Code Integration

### Memory Files
This repository uses Claude Code memory files for development workflow:

- **CLAUDE.md** (this file): Fork-specific documentation, patches, and setup instructions (committed to git)
- **CLAUDE.local.md**: Machine-specific settings like conda environment names (gitignored)

When using Claude Code's `#` command:
- If `CLAUDE.local.md` exists, it will be updated with new memories
- If only `CLAUDE.md` exists, it will be updated instead
- Always check `CLAUDE.local.md` first for local environment configuration

### Best Practices
- Keep environment-specific instructions in `CLAUDE.local.md`
- Use this `CLAUDE.md` for fork-wide documentation and patch instructions
- When cloning to a new machine, create `CLAUDE.local.md` with your local conda environment name

## 📝 Notes

- This fork tracks upstream `v0.4.1+` with the new `src/` layout
- **Apple MPS support is built into upstream v0.4.1+** - no patches needed!
- TensorBoard patches are designed to be robust against upstream changes
- Keep patch files updated if you modify the TensorBoard features
- All enhancements are backward-compatible with standard LeRobot usage
- See `PATCHES_README.md` for detailed patch documentation

## 📚 Pi0 Flow Matching Code Walkthrough

### Overview
Pi0 implements Flow Matching for robot action generation, combining vision-language understanding (PaliGemma) with continuous action synthesis through learned velocity fields.

### Core Flow Matching Theory → Code Mapping

**Theoretical Foundation:**
Flow Matching learns a velocity field `v_θ(x_t, t)` to transform noise to data through an ODE:
```
dx/dt = v_θ(x_t, t)
```

**Pi0 Implementation** (`src/lerobot/policies/pi0/modeling_pi0.py:716-755`):

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    # Sample noise and time if not provided
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)  # ε ~ N(0,I)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)  # t ~ Beta(1.5, 1.0)
    
    # Flow Matching interpolation: x_t = t * noise + (1-t) * actions
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    
    # Target velocity field: u_t = noise - actions (for linear interpolation)
    u_t = noise - actions
    
    # Forward through neural network to predict velocity field v_t
    # ... (vision-language processing)
    v_t = self.action_out_proj(suffix_out)
    
    # Flow matching loss: ||u_t - v_t||²
    losses = F.mse_loss(u_t, v_t, reduction="none")
    return losses
```

### Key Components

#### Noise and Time Sampling
```python
def sample_noise(self, shape, device):
    # Standard Gaussian noise ε ~ N(0, I)
    noise = torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
    return noise

def sample_time(self, bsize, device):
    # Beta distribution for time sampling: t ~ Beta(1.5, 1.0)
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
    time = time_beta * 0.999 + 0.001  # Avoid exact 0 and 1
    return time
```

**Why Beta(1.5, 1.0)?** Biases sampling toward larger time values, focusing training on harder denoising steps.

#### Linear Interpolation Path
```python
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions  # Linear interpolation
u_t = noise - actions  # Target velocity field for linear path
```

This implements **conditional flow matching** with linear interpolation:
- At `t=0`: `x_t = actions` (clean data)
- At `t=1`: `x_t = noise` (pure noise)
- Target velocity `u_t = dx_t/dt = noise - actions` (constant for linear paths)

#### Vision-Language Processing
```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    # Process images with SigLIP vision encoder
    img_emb = self.paligemma_with_expert.embed_image(img)
    img_emb = img_emb * torch.tensor(img_emb_dim**0.5, ...)  # Normalization
    
    # Process language with Gemma embeddings
    lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)  # Normalization
    
    return embs, pad_masks, att_masks
```

#### Action-Time Conditioning
```python
def embed_suffix(self, state, noisy_actions, timestep):
    # Sinusoidal time embedding
    time_emb = create_sinusoidal_pos_embedding(
        timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
    )
    
    # Fuse time + action through MLP
    action_emb = self.action_in_proj(noisy_actions)
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)
    action_time_emb = self.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = self.action_time_mlp_out(action_time_emb)
    
    return embs, pad_masks, att_masks
```

### Inference: ODE Solving
```python
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
    # Start from pure noise
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    dt = -1.0 / self.config.num_steps  # Negative step (going backward in time)
    
    # Euler integration to solve ODE: dx/dt = v_θ(x_t, t)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        
        # Euler step: x_{t+dt} = x_t + dt * v_t
        x_t += dt * v_t
        time += dt
    
    return x_t  # Final denoised actions
```

**Key Points:**
- **Backward integration**: Start at `t=1` (noise) → end at `t=0` (clean actions)
- **Euler method**: Simple first-order ODE solver
- **10 steps default**: Balance between quality and speed

### Pi0 vs Standard Flow Matching Differences

1. **Multi-modal Conditioning**: Conditions on vision + language + robot state
2. **Action Chunking**: Predicts sequences of 50 actions simultaneously
3. **Robot-Specific Adaptations**: Special handling for Aloha robots
4. **Attention Architecture**: Uses transformer attention rather than simple MLPs

### Training vs Inference Flow

**Training** (`forward` method):
1. Sample random `t ~ Beta(1.5, 1.0)` and `ε ~ N(0,I)`
2. Create noisy actions: `x_t = t·ε + (1-t)·actions`
3. Predict velocity: `v_t = f_θ(x_t, t, vision, language, state)`
4. Compute loss: `L = ||ε - actions - v_t||²`

**Inference** (`sample_actions` method):
1. Start with pure noise: `x_1 = ε ~ N(0,I)`
2. Iteratively denoise with Euler steps: `x_{t-dt} = x_t - dt·v_θ(x_t, t, ...)`
3. Return final actions: `x_0`

### Key Files
- `src/lerobot/policies/pi0/modeling_pi0.py` - Main Flow Matching implementation
- `src/lerobot/policies/pi0/configuration_pi0.py` - Configuration
- `src/lerobot/policies/pi0/paligemma_with_expert.py` - Vision-language model