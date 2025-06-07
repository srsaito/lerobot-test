# ACT Policy Action Landscape Visualization

This repository contains tools for visualizing Action Chunking Transformer (ACT) policy action landscapes in the PushT simulation environment. The visualization creates side-by-side videos showing:

- **Left side**: Expert demonstration replay in the PushT environment
- **Right side**: Action space heatmap showing policy scores for all possible actions at each timestep

## Features

### Basic Visualization (`visualize_act_policy_landscape.py`)
- Side-by-side environment and action landscape visualization
- Configurable action space resolution
- Expert action marking with red crosses
- Policy score heatmaps using L1 loss

### Advanced Visualization (`visualize_act_landscape_advanced.py`)
- Multiple scoring methods:
  - **L1 Loss**: Uses negative L1 loss as policy score
  - **Confidence**: Based on action prediction consistency (negative variance)
  - **Likelihood**: Action likelihood under the policy's learned distribution
- Policy prediction visualization with arrows
- Confidence interval contours
- Customizable colormaps
- Batch processing for efficiency
- Memory-optimized evaluation

## Installation

First, ensure you have the LeRobot environment set up with PushT support:

```bash
pip install -e ".[pusht]"
```

Additional dependencies for visualization:
```bash
pip install matplotlib opencv-python imageio tqdm
```

## Usage

### Quick Start

```bash
# Basic visualization
python visualize_act_policy_landscape.py \
    --policy_path lerobot/act_pusht \
    --episode_idx 0 \
    --max_steps 100

# Advanced visualization with confidence scoring
python visualize_act_landscape_advanced.py \
    --policy_path lerobot/act_pusht \
    --episode_idx 0 \
    --scoring_method confidence \
    --show_policy_prediction true \
    --colormap plasma \
    --max_steps 100
```

### Example Script

Run multiple visualizations with different configurations:

```bash
python example_run_act_landscape_viz.py
```

This will generate visualizations with different scoring methods and episodes.

## Parameters

### Common Parameters

- `--policy_path`: Path to trained ACT policy (HuggingFace model or local checkpoint)
- `--dataset_repo_id`: HuggingFace dataset repository ID (default: "lerobot/pusht")
- `--episode_idx`: Episode index to visualize (default: 0)
- `--output_dir`: Output directory for videos
- `--action_resolution`: Resolution for action space sampling (NxN grid, default: 50)
- `--max_steps`: Maximum number of steps to visualize
- `--device`: Device to run policy on (default: "cuda")

### Advanced Parameters

- `--scoring_method`: Method for scoring actions
  - `l1_loss`: Negative L1 loss (lower loss = higher score)
  - `confidence`: Negative variance of action predictions
  - `likelihood`: Log likelihood under policy distribution
- `--colormap`: Matplotlib colormap for heatmap (e.g., "viridis", "plasma", "hot")
- `--show_policy_prediction`: Show policy's predicted action with arrows
- `--show_confidence_intervals`: Show contours for high-confidence regions
- `--batch_size`: Batch size for efficient action evaluation (default: 100)
- `--target_height`: Target height for video frames (default: 480)

## Output Files

Each visualization generates:

1. **Environment video**: `episode_{idx}_environment.mp4`
   - Pure environment replay
   
2. **Heatmap video**: `episode_{idx}_heatmaps_{scoring_method}.mp4`
   - Action landscape heatmaps only
   
3. **Combined video**: `episode_{idx}_combined_{scoring_method}.mp4`
   - Side-by-side environment and heatmap
   
4. **Configuration file**: `episode_{idx}_config.json`
   - Visualization parameters and statistics

## Understanding the Visualization

### Heatmap Interpretation

- **Color intensity**: Represents policy score (higher = better according to policy)
- **Red cross (+)**: Expert action at current timestep
- **White star (*)**: Policy's predicted action (if enabled)
- **Yellow arrow**: Direction from policy prediction to expert action
- **White contours**: High-confidence regions (top 10% of scores)

### Scoring Methods

1. **L1 Loss**: Shows how much the policy "likes" each action based on training loss
2. **Confidence**: Shows where the policy is most certain about its predictions
3. **Likelihood**: Shows the probability of each action under the policy's learned distribution

## Example Use Cases

### 1. Policy Debugging
```bash
python visualize_act_landscape_advanced.py \
    --policy_path outputs/train/my_act_model/checkpoints/001000/pretrained_model \
    --scoring_method confidence \
    --show_policy_prediction true \
    --episode_idx 0
```

### 2. Comparing Scoring Methods
```bash
# Run with different scoring methods
for method in l1_loss confidence likelihood; do
    python visualize_act_landscape_advanced.py \
        --policy_path lerobot/act_pusht \
        --scoring_method $method \
        --output_dir outputs/comparison_$method \
        --episode_idx 0
done
```

### 3. High-Resolution Analysis
```bash
python visualize_act_landscape_advanced.py \
    --policy_path lerobot/act_pusht \
    --action_resolution 100 \
    --scoring_method likelihood \
    --show_confidence_intervals true \
    --max_steps 50
```

## Performance Tips

1. **Action Resolution**: Start with 30-50 for quick tests, use 100+ for detailed analysis
2. **Batch Size**: Increase to 200-500 if you have more GPU memory
3. **Max Steps**: Limit to 50-100 steps for initial exploration
4. **Device**: Use CUDA if available for faster processing

## Troubleshooting

### Memory Issues
- Reduce `--action_resolution` (e.g., from 50 to 30)
- Decrease `--batch_size` (e.g., from 100 to 50)
- Use `--device cpu` if GPU memory is limited

### Policy Loading Issues
- Ensure the policy path is correct
- For HuggingFace models, check internet connection
- For local checkpoints, verify the path contains `config.json` and `model.safetensors`

### Environment Issues
- Install PushT support: `pip install -e ".[pusht]"`
- Check that `gym_pusht` is properly installed

## Technical Details

### Action Space Sampling
The visualization samples actions uniformly across the 2D action space (x, y movements) and evaluates each action's score according to the policy. The action space bounds are automatically extracted from the PushT environment.

### Policy Scoring
Different scoring methods provide different insights:
- **L1 Loss**: Direct training objective feedback
- **Confidence**: Model certainty/uncertainty
- **Likelihood**: Probabilistic action preferences

### Memory Optimization
The advanced visualizer processes actions in batches to handle large action grids efficiently while staying within GPU memory limits.

## Citation

If you use this visualization tool in your research, please cite the original ACT paper:

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
``` 