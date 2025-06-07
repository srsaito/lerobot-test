#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch

def debug_grid_arrangement():
    """Debug how our action grid and score grid are arranged."""
    
    print("=== Debugging Grid Arrangement ===")
    
    # Simulate the same grid creation as our visualization
    action_low = np.array([0., 0.])
    action_high = np.array([512., 512.])
    action_resolution = 5  # Small for debugging
    
    print(f"Action space: {action_low} to {action_high}")
    print(f"Action resolution: {action_resolution}x{action_resolution}")
    
    # Create action grid the same way as the main script
    x_actions = np.linspace(action_low[0], action_high[0], action_resolution)
    y_actions = np.linspace(action_low[1], action_high[1], action_resolution)
    
    print(f"X actions: {x_actions}")
    print(f"Y actions: {y_actions}")
    
    xx, yy = np.meshgrid(x_actions, y_actions)
    actions = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    print(f"\nAction grid shape: {actions.shape}")
    print("First few actions:")
    for i in range(min(10, len(actions))):
        print(f"  Action {i}: {actions[i]}")
    
    # Create a simple score pattern to understand the mapping
    # Give higher scores to actions with lower Y values (should appear at top)
    scores = []
    for action in actions:
        x, y = action
        # Higher score for lower Y values (should appear at top of environment)
        score = 512 - y  # Low Y gets high score
        scores.append(score)
    
    scores = np.array(scores)
    
    # Reshape to 2D grid (same as main script)
    score_grid = scores.reshape(action_resolution, action_resolution)
    
    print(f"\nScore grid shape: {score_grid.shape}")
    print("Score grid:")
    print(score_grid)
    print(f"Row 0 (should be actions with Y={y_actions[0]}): {score_grid[0, :]}")
    print(f"Row -1 (should be actions with Y={y_actions[-1]}): {score_grid[-1, :]}")
    
    # Test both origins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Test action positions to understand mapping
    test_actions = [
        [256, y_actions[0]],   # Low Y (should appear at top of environment)
        [256, y_actions[-1]],  # High Y (should appear at bottom of environment)
    ]
    
    extent = [action_low[0], action_high[0], action_low[1], action_high[1]]
    
    # Origin = 'upper'
    im1 = ax1.imshow(score_grid, extent=extent, origin='upper', cmap='viridis')
    for i, action in enumerate(test_actions):
        color = 'red' if i == 0 else 'blue'
        ax1.plot(action[0], action[1], '+', color=color, markersize=15, markeredgewidth=3,
                label=f'Y={action[1]:.1f}')
    ax1.set_title("Origin='upper'")
    ax1.set_xlabel('Action X')
    ax1.set_ylabel('Action Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Origin = 'lower'
    im2 = ax2.imshow(score_grid, extent=extent, origin='lower', cmap='viridis')
    for i, action in enumerate(test_actions):
        color = 'red' if i == 0 else 'blue'
        ax2.plot(action[0], action[1], '+', color=color, markersize=15, markeredgewidth=3,
                label=f'Y={action[1]:.1f}')
    ax2.set_title("Origin='lower'")
    ax2.set_xlabel('Action X')
    ax2.set_ylabel('Action Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_grid_arrangement.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Analysis ===")
    print(f"In our score pattern, low Y actions have high scores (bright colors)")
    print(f"Low Y actions should appear at TOP of environment")
    print(f"Therefore, bright colors should appear at TOP of heatmap")
    print(f"\nWith origin='upper':")
    print(f"  - Row 0 of score_grid appears at TOP")
    print(f"  - Row 0 corresponds to Y={y_actions[0]} (low Y)")
    print(f"  - This is CORRECT if low Y should appear at top")
    print(f"\nWith origin='lower':")
    print(f"  - Row 0 of score_grid appears at BOTTOM") 
    print(f"  - Row 0 corresponds to Y={y_actions[0]} (low Y)")
    print(f"  - This is INCORRECT if low Y should appear at top")

if __name__ == "__main__":
    debug_grid_arrangement() 