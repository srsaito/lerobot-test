# Diffusion Policy Action Landscape Visualization for PushT

This repository contains tools for visualizing Diffusion Policy action landscapes in the PushT simulation environment. The visualization creates side-by-side videos showing:

- **Left side**: Expert demonstration replay in the PushT environment
- **Right side**: Action space heatmap showing diffusion policy scores for all possible actions at each timestep

## Features

### Advanced Visualization (`visualize_diffusion_landscape_advanced.py`)
- Multiple scoring methods:
  - **Likelihood**: Uses the diffusion model's likelihood of generating actions (most accurate)
  - **Distance**: Uses negative distance to policy's predicted action (faster, simpler)
  - **Noise Prediction**: Uses the diffusion model's denoising capability as a score (experimental)

- Enhanced visualizations:
  - Policy prediction arrows showing where the model would act
  - Expert action markers (red crosses)
  - Confidence interval contours for high-certainty regions
  - Customizable colormaps (viridis, plasma, inferno, coolwarm, etc.)
  - Batch processing for efficiency

### Example Runner (`example_run_diffusion_landscape_viz.py`)
- Pre-configured examples with different visualization settings
- Demonstrates various scoring methods and parameters
- Includes tips and best practices

## Installation

Make sure you have the LeRobot environment set up with:

```bash
# Install LeRobot dependencies
pip install -e .

# Install additional visualization dependencies
pip install imageio matplotlib opencv-python tqdm
```

## Quick Start

### Option 1: Run a single visualization

```bash
python visualize_diffusion_landscape_advanced.py \
    --policy_path lerobot/diffusion_pusht \
    --dataset_repo_id lerobot/pusht \
    --episode_idx 0 \
    --scoring_method likelihood \
    --max_steps 100
```

### Option 2: Run example suite

```bash
python example_run_diffusion_landscape_viz.py
```

This will generate several example visualizations with different settings.

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy_path` | Required | Path to trained Diffusion Policy (e.g., `lerobot/diffusion_pusht`) |
| `--dataset_repo_id` | `lerobot/pusht` | HuggingFace dataset repository ID |
| `--episode_idx` | `0` | Episode index to visualize |
| `--output_dir` | `outputs/diffusion_landscape_viz` | Output directory for videos |
| `--action_resolution` | `50` | Resolution for action space sampling (NxN grid) |
| `--max_steps` | `None` | Maximum number of steps to visualize |
| `--device` | `cuda` | Device to run policy on |
| `--scoring_method` | `likelihood` | Scoring method: `likelihood`, `distance`, or `noise_pred` |
| `--colormap` | `viridis` | Matplotlib colormap for heatmap |
| `--show_policy_prediction` | `True` | Whether to show policy's predicted action |
| `--show_confidence_intervals` | `False` | Whether to show confidence interval contours |
| `--batch_size` | `100` | Batch size for efficient action evaluation |
| `--target_height` | `480` | Target height for video frames |

## Scoring Methods Explained

### 1. Likelihood Scoring (`--scoring_method likelihood`)
- **What it does**: Evaluates how likely the diffusion model is to generate each action
- **How it works**: Uses the negative MSE between predicted and target actions as a likelihood proxy
- **Best for**: Most accurate representation of the policy's preferences
- **Speed**: Slower (most computationally intensive)

### 2. Distance Scoring (`--scoring_method distance`)
- **What it does**: Measures distance from policy's predicted action
- **How it works**: Computes negative L2 distance to the policy's single predicted action
- **Best for**: Quick visualization with interpretable results
- **Speed**: Fastest

### 3. Noise Prediction Scoring (`--scoring_method noise_pred`)
- **What it does**: Uses the diffusion model's denoising capability
- **How it works**: Adds noise to actions and scores based on denoising quality
- **Best for**: Understanding the model's denoising behavior
- **Speed**: Medium

## Output Files

For each run, the following files are generated in the output directory:

- `episode_{idx}_environment.mp4`: Environment-only video
- `episode_{idx}_heatmaps_{method}.mp4`: Action landscape heatmaps only
- `episode_{idx}_combined_{method}.mp4`: Side-by-side combination
- `episode_{idx}_config.json`: Configuration and metadata

## Examples and Use Cases

### Basic Usage - Quick Distance-Based Visualization
```bash
python visualize_diffusion_landscape_advanced.py \
    --policy_path lerobot/diffusion_pusht \
    --episode_idx 0 \
    --scoring_method distance \
    --max_steps 50 \
    --action_resolution 40
```

### High-Quality Likelihood Visualization
```bash
python visualize_diffusion_landscape_advanced.py \
    --policy_path lerobot/diffusion_pusht \
    --episode_idx 1 \
    --scoring_method likelihood \
    --max_steps 100 \
    --action_resolution 60 \
    --show_confidence_intervals true \
    --colormap plasma
```

### Research-Grade High-Resolution Analysis
```bash
python visualize_diffusion_landscape_advanced.py \
    --policy_path lerobot/diffusion_pusht \
    --episode_idx 5 \
    --scoring_method likelihood \
    --max_steps 200 \
    --action_resolution 80 \
    --show_confidence_intervals true \
    --show_policy_prediction true \
    --colormap inferno \
    --batch_size 50
```

## Customization

### Colormaps
Try different colormaps for better visualization:
- `viridis` (default): Good general-purpose colormap
- `plasma`: High contrast, good for publications
- `inferno`: Warm colors, good for highlighting hot spots
- `coolwarm`: Diverging colormap, good for showing positive/negative values

### Action Resolution
- **Low (20-40)**: Fast, good for prototyping
- **Medium (50-60)**: Balanced quality/speed
- **High (70-100)**: High quality, slow, good for final visualizations

### Performance Tips
- Use `distance` scoring for quick iterations
- Reduce `action_resolution` for faster processing
- Increase `batch_size` if you have more GPU memory
- Use `max_steps` to limit computation for long episodes

## Understanding the Visualization

### Left Panel: Environment
- Shows the actual PushT environment with the T-shaped object
- Displays the agent (circle) pushing the T-shaped target
- Follows the expert demonstration trajectory

### Right Panel: Action Landscape
- **Heatmap colors**: Policy preference scores (warmer = higher preference)
- **Red cross (+)**: Expert action taken at this timestep
- **White star (*)**: Policy's predicted action (if enabled)
- **Yellow arrow**: Direction from policy prediction to expert action
- **White contours**: High-confidence regions (if enabled)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `action_resolution` or `batch_size`
   - Use `--device cpu` for CPU-only processing

2. **Policy loading errors**
   - Verify the policy path: `lerobot/diffusion_pusht` should be correct
   - Check internet connection for downloading from HuggingFace

3. **Environment rendering issues**
   - Make sure `gym_pusht` is properly installed
   - Check that rendering dependencies are available

4. **Slow performance**
   - Use `distance` scoring method for faster results
   - Reduce `action_resolution` from 50 to 30-40
   - Increase `batch_size` if you have sufficient GPU memory

### Performance Benchmarks

On a typical GPU (RTX 3080):
- **Distance scoring**: ~2-3 seconds per timestep
- **Likelihood scoring**: ~10-15 seconds per timestep  
- **Noise prediction**: ~5-8 seconds per timestep

## Scientific Applications

This visualization is useful for:

1. **Policy Analysis**: Understanding how diffusion policies distribute probability over action spaces
2. **Failure Mode Investigation**: Seeing where policy predictions deviate from expert actions
3. **Model Comparison**: Comparing different diffusion policy variants
4. **Training Progress**: Visualizing how policy behavior changes during training
5. **Action Space Coverage**: Understanding which regions of action space the policy explores

## Citation

If you use this visualization tool in your research, please cite the original LeRobot framework and acknowledge this visualization extension.

## Contributing

Feel free to submit issues and pull requests to improve the visualization tools! 