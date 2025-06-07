#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_pusht

def test_cross_with_different_origins():
    """Test how the red cross appears with different matplotlib origin settings."""
    
    # Create environment
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )
    
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    # Test with a high Y action (should move agent UP in environment)
    test_action = np.array([256, 450])  # High Y value
    
    # Get environment visual
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(test_action)
    env_frame = env.render()
    
    # Create comparison plots with different origins
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show environment
    axes[0].imshow(env_frame)
    axes[0].set_title(f'Environment\nAction: {test_action}')
    axes[0].axis('off')
    
    # Create dummy heatmap data
    heatmap_data = np.random.rand(20, 20)
    extent = [action_low[0], action_high[0], action_low[1], action_high[1]]
    
    # Heatmap with origin='lower' (current approach)
    axes[1].imshow(heatmap_data, extent=extent, origin='lower', cmap='viridis', alpha=0.7)
    axes[1].plot(test_action[0], test_action[1], 'r+', markersize=20, markeredgewidth=4)
    axes[1].set_title("Origin='lower'\n(Current approach)")
    axes[1].set_xlabel('Action X')
    axes[1].set_ylabel('Action Y')
    axes[1].grid(True, alpha=0.3)
    
    # Heatmap with origin='upper' (alternative)
    axes[2].imshow(heatmap_data, extent=extent, origin='upper', cmap='viridis', alpha=0.7)
    axes[2].plot(test_action[0], test_action[1], 'r+', markersize=20, markeredgewidth=4)
    axes[2].set_title("Origin='upper'\n(Alternative approach)")
    axes[2].set_xlabel('Action X')
    axes[2].set_ylabel('Action Y')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_origin_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Test action: {test_action}")
    print(f"Action bounds: low={action_low}, high={action_high}")
    print("Generated: cross_origin_comparison.png")
    
    # Now let's test extreme cases more clearly
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test two extreme Y actions
    low_y_action = np.array([256, 50])   # Low Y
    high_y_action = np.array([256, 450])  # High Y
    
    test_actions = [low_y_action, high_y_action]
    labels = ['Low Y (50)', 'High Y (450)']
    
    for i, (action, label) in enumerate(zip(test_actions, labels)):
        # Get environment result
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(action)
        frame = env.render()
        
        # Show environment
        axes[i, 0].imshow(frame)
        axes[i, 0].set_title(f'Environment: {label}')
        axes[i, 0].axis('off')
        
        # Show heatmap position with origin='lower'
        axes[i, 1].imshow(heatmap_data, extent=extent, origin='lower', cmap='viridis', alpha=0.7)
        axes[i, 1].plot(action[0], action[1], 'r+', markersize=20, markeredgewidth=4)
        axes[i, 1].set_title(f'Heatmap: {label}')
        axes[i, 1].set_xlabel('Action X')
        axes[i, 1].set_ylabel('Action Y')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('extreme_y_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generated: extreme_y_test.png")
    print("\nAnalysis:")
    print("- Low Y action (50): Should move agent toward BOTTOM of environment")
    print("- High Y action (450): Should move agent toward TOP of environment")
    print("- With origin='lower': low Y values appear at bottom of heatmap")
    print("- With origin='lower': high Y values appear at top of heatmap")
    print("- This should create intuitive correspondence")

if __name__ == "__main__":
    test_cross_with_different_origins() 