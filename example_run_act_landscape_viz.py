#!/usr/bin/env python

"""
Example script demonstrating how to use the ACT Policy Landscape Visualization tools.

This script shows several different ways to run the visualization with different parameters
and scoring methods.
"""

import subprocess
import sys
from pathlib import Path

def run_visualization(
    policy_path: str,
    dataset_repo_id: str = "lerobot/pusht",
    episode_idx: int = 0,
    scoring_method: str = "confidence",
    action_resolution: int = 50,
    max_steps: int = 100,
    output_dir: str = None,
    advanced: bool = True
):
    """Run the ACT policy landscape visualization."""
    
    if output_dir is None:
        output_dir = f"outputs/act_viz_{scoring_method}_ep{episode_idx}"
    
    script_name = "visualize_act_landscape_advanced.py" if advanced else "visualize_act_policy_landscape.py"
    
    cmd = [
        sys.executable, script_name,
        "--policy_path", policy_path,
        "--dataset_repo_id", dataset_repo_id,
        "--episode_idx", str(episode_idx),
        "--output_dir", output_dir,
        "--action_resolution", str(action_resolution),
        "--max_steps", str(max_steps),
        "--scoring_method", scoring_method,
    ]
    
    if advanced:
        cmd.extend([
            "--show_policy_prediction", "true",
            "--colormap", "plasma",
        ])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    """Run multiple visualization examples."""
    
    # You need to provide a path to a trained ACT policy
    # This could be a HuggingFace model or a local checkpoint
    policy_path = "lerobot/act_pusht"  # Replace with your actual policy path
    
    # Check if policy path exists (for local paths)
    if not policy_path.startswith("lerobot/") and not Path(policy_path).exists():
        print(f"Error: Policy path {policy_path} does not exist.")
        print("Please provide a valid path to a trained ACT policy.")
        print("Examples:")
        print("  - HuggingFace model: 'lerobot/act_pusht'")
        print("  - Local checkpoint: 'outputs/train/act_pusht/checkpoints/001000/pretrained_model'")
        return
    
    print("=" * 80)
    print("ACT Policy Landscape Visualization Examples")
    print("=" * 80)
    
    examples = [
        {
            "name": "Basic Confidence Visualization",
            "scoring_method": "confidence",
            "episode_idx": 0,
            "action_resolution": 30,  # Lower resolution for faster execution
            "max_steps": 50,
        },
        {
            "name": "L1 Loss Visualization",
            "scoring_method": "l1_loss",
            "episode_idx": 1,
            "action_resolution": 40,
            "max_steps": 75,
        },
        {
            "name": "Likelihood Visualization",
            "scoring_method": "likelihood",
            "episode_idx": 2,
            "action_resolution": 35,
            "max_steps": 60,
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Running: {example['name']}")
        print("-" * 50)
        
        try:
            run_visualization(
                policy_path=policy_path,
                episode_idx=example["episode_idx"],
                scoring_method=example["scoring_method"],
                action_resolution=example["action_resolution"],
                max_steps=example["max_steps"],
                output_dir=f"outputs/act_viz_examples/{example['scoring_method']}_ep{example['episode_idx']}",
                advanced=True
            )
            print(f"✅ {example['name']} completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {example['name']} failed with error: {e}")
            continue
        except Exception as e:
            print(f"❌ {example['name']} failed with unexpected error: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print("Check the 'outputs/act_viz_examples/' directory for results.")
    print("=" * 80)

if __name__ == "__main__":
    main() 