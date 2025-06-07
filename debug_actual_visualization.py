#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_pusht
import torch
from pathlib import Path

# Import from our visualization scripts
from visualize_diffusion_landscape_advanced import AdvancedDiffusionPolicyLandscapeVisualizer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def debug_actual_visualization():
    """Debug the actual visualization to see what's happening with expert actions."""
    
    print("=== Debugging Actual Visualization ===")
    
    # Load dataset to get real expert actions
    dataset = LeRobotDataset("lerobot/pusht")
    
    # Get episode data for episode 0
    from_idx = dataset.episode_data_index["from"][0].item()
    to_idx = dataset.episode_data_index["to"][0].item()
    episode_data = dataset.hf_dataset.select(range(from_idx, min(from_idx + 5, to_idx)))  # Just first 5 steps
    
    expert_actions = np.array(episode_data["action"])
    print(f"First 5 expert actions from dataset:")
    for i, action in enumerate(expert_actions):
        print(f"  Step {i}: {action}")
    
    # Create environment to test these actions
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )
    
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"\nEnvironment action bounds: low={action_low}, high={action_high}")
    
    # Test what these expert actions actually do in the environment
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(min(3, len(expert_actions))):
        expert_action = expert_actions[i]
        
        # Reset environment and step with expert action
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(expert_action)
        env_frame = env.render()
        
        # Show environment result
        axes[0, i].imshow(env_frame)
        axes[0, i].set_title(f'Step {i}: Env Result\nAction: ({expert_action[0]:.1f}, {expert_action[1]:.1f})')
        axes[0, i].axis('off')
        
        # Show where this would appear on heatmap with our current settings
        dummy_heatmap = np.zeros((50, 50))
        im = axes[1, i].imshow(
            dummy_heatmap,
            extent=[action_low[0], action_high[0], action_low[1], action_high[1]],
            origin='lower',  # Current setting
            cmap='viridis',
            alpha=0.5
        )
        
        # Plot the red cross
        axes[1, i].plot(expert_action[0], expert_action[1], 'r+', markersize=20, markeredgewidth=4)
        
        axes[1, i].set_title(f'Heatmap Position\nY={expert_action[1]:.1f}')
        axes[1, i].set_xlabel('Action X')
        axes[1, i].set_ylabel('Action Y')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add some reference lines
        axes[1, i].axhline(y=expert_action[1], color='red', linestyle='--', alpha=0.5)
        axes[1, i].axvline(x=expert_action[0], color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('debug_actual_expert_actions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generated: debug_actual_expert_actions.png")
    
    # Now let's manually create the same visualization function as our code does
    print("\n=== Testing Visualization Function Directly ===")
    
    # Create a simple test using our visualization approach
    test_action = expert_actions[0]  # Use first expert action
    
    # Create environment and get observation
    obs, _ = env.reset()
    env_frame_before = env.render()
    
    # Create our heatmap the same way the visualization does
    action_resolution = 20  # Smaller for debugging
    action_grid = torch.zeros((action_resolution * action_resolution, 2))
    
    # Create action grid
    x_values = np.linspace(action_low[0], action_high[0], action_resolution)
    y_values = np.linspace(action_low[1], action_high[1], action_resolution)
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            idx = i * action_resolution + j
            action_grid[idx, 0] = float(x)  # Convert to float
            action_grid[idx, 1] = float(y)  # Convert to float
    
    # Create dummy scores (random for visualization)
    scores = torch.rand(action_resolution * action_resolution)
    score_grid = scores.numpy().reshape(action_resolution, action_resolution)
    
    # Now create the visualization exactly as our code does
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Environment before action
    ax1.imshow(env_frame_before)
    ax1.set_title('Environment (Before Action)')
    ax1.axis('off')
    
    # Heatmap with red cross (exactly as our code does it)
    im = ax2.imshow(
        score_grid,
        extent=[action_low[0], action_high[0], action_low[1], action_high[1]],
        origin='lower',  # This is our current setting
        cmap='viridis',
        aspect='auto',
        alpha=0.8
    )
    
    # Mark expert action with red cross (exactly as our code does it)
    ax2.plot(test_action[0], test_action[1], 'r+', markersize=20, markeredgewidth=4, 
            label='Expert Action')
    
    ax2.set_xlabel('Action X')
    ax2.set_ylabel('Action Y')
    ax2.set_title(f'Action Heatmap\nExpert Action: ({test_action[0]:.1f}, {test_action[1]:.1f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_visualization_function.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generated: debug_visualization_function.png")
    print(f"Test expert action: {test_action}")
    
    # Check if the action seems reasonable
    if test_action[1] < (action_high[1] - action_low[1]) / 2:
        print("This is a LOW Y action - cross should appear in BOTTOM half of heatmap")
    else:
        print("This is a HIGH Y action - cross should appear in TOP half of heatmap")

if __name__ == "__main__":
    debug_actual_visualization() 