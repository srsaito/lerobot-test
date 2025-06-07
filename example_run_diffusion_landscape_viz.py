#!/usr/bin/env python

"""
Example script demonstrating how to use the Diffusion Policy Landscape Visualization tools.

This script shows several different ways to run the visualization with different parameters
and scoring methods for the PushT environment.
"""

import subprocess
import sys
from pathlib import Path

def run_visualization(
    policy_path: str = "lerobot/diffusion_pusht",
    dataset_repo_id: str = "lerobot/pusht",
    episode_idx: int = 0,
    scoring_method: str = "likelihood",
    action_resolution: int = 50,
    max_steps: int = 100,
    output_dir: str = None,
    device: str = "cpu",
    show_policy_prediction: bool = True,
    show_confidence_intervals: bool = False,
):
    """Run the Diffusion Policy landscape visualization."""
    
    if output_dir is None:
        output_dir = f"outputs/diffusion_viz_{scoring_method}_ep{episode_idx}"
    
    script_name = "visualize_diffusion_landscape_advanced.py"
    
    cmd = [
        "python", script_name,
        "--policy_path", policy_path,
        "--dataset_repo_id", dataset_repo_id,
        "--episode_idx", str(episode_idx),
        "--scoring_method", scoring_method,
        "--action_resolution", str(action_resolution),
        "--max_steps", str(max_steps),
        "--output_dir", output_dir,
        "--device", device,
        "--show_policy_prediction", str(show_policy_prediction).lower(),
        "--show_confidence_intervals", str(show_confidence_intervals).lower(),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully generated visualization in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running visualization: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: {script_name} not found. Make sure it's in the current directory.")
        return False
    
    return True

def main():
    """Run several example visualizations."""
    
    print("🚀 Diffusion Policy Action Landscape Visualization Examples")
    print("=" * 60)
    
    # Example 1: Basic likelihood-based visualization
    print("\n📊 Example 1: Basic likelihood visualization")
    success = run_visualization(
        episode_idx=0,
        scoring_method="likelihood",
        max_steps=50,
        action_resolution=40,
        output_dir="outputs/diffusion_basic_likelihood"
    )
    
    if not success:
        print("Failed to run basic example. Please check your setup.")
        return
    
    # Example 2: Distance-based scoring (faster, simpler)
    print("\n📊 Example 2: Distance-based scoring")
    run_visualization(
        episode_idx=1,
        scoring_method="distance",
        max_steps=50,
        action_resolution=40,
        output_dir="outputs/diffusion_distance_scoring"
    )
    
    # Example 3: High-resolution with confidence intervals
    print("\n📊 Example 3: High-resolution with confidence intervals")
    run_visualization(
        episode_idx=2,
        scoring_method="likelihood",
        max_steps=30,
        action_resolution=60,
        show_confidence_intervals=True,
        output_dir="outputs/diffusion_high_res_confidence"
    )
    
    # Example 4: Noise prediction scoring
    print("\n📊 Example 4: Noise prediction scoring")
    run_visualization(
        episode_idx=0,
        scoring_method="noise_pred",
        max_steps=40,
        action_resolution=40,
        output_dir="outputs/diffusion_noise_prediction"
    )
    
    print("\n🎉 All examples completed!")
    print("\nGenerated outputs:")
    print("  - outputs/diffusion_basic_likelihood/")
    print("  - outputs/diffusion_distance_scoring/")
    print("  - outputs/diffusion_high_res_confidence/")
    print("  - outputs/diffusion_noise_prediction/")
    
    print("\n📚 Tips:")
    print("  - Use 'likelihood' scoring for most accurate policy visualization")
    print("  - Use 'distance' scoring for faster generation with similar results")
    print("  - Increase action_resolution for smoother heatmaps (but slower)")
    print("  - Try different episodes (episode_idx) for variety")
    print("  - Experiment with different colormaps: viridis, plasma, inferno, coolwarm")

if __name__ == "__main__":
    main() 