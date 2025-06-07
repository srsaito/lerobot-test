#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_pusht  # noqa: F401
import logging

logging.basicConfig(level=logging.INFO)

def test_coordinate_alignment():
    """Test coordinate alignment between PushT environment and action heatmap."""
    
    # Create PushT environment using the correct gym registration
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )
    
    # Reset environment to get initial state
    obs, _ = env.reset()
    
    # Get action bounds
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"Action space bounds: low={action_low}, high={action_high}")
    
    # Test specific actions at corners and center
    test_actions = [
        (action_low[0], action_low[1]),     # Bottom-left corner
        (action_high[0], action_low[1]),    # Bottom-right corner  
        (action_low[0], action_high[1]),    # Top-left corner
        (action_high[0], action_high[1]),   # Top-right corner
        ((action_low[0] + action_high[0])/2, (action_low[1] + action_high[1])/2),  # Center
    ]
    
    labels = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right", "Center"]
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (action, label) in enumerate(zip(test_actions, labels)):
        # Reset environment
        obs, _ = env.reset()
        
        # Step with the test action
        obs, _, _, _, _ = env.step(np.array(action))
        
        # Get environment frame
        env_frame = env.render()
        
        # Show environment state on left
        if i == 0:  # Only show environment once
            ax1.imshow(env_frame)
            ax1.set_title("PushT Environment")
            ax1.set_xlabel("Pixel X (left-right)")
            ax1.set_ylabel("Pixel Y (top-bottom)")
        
        # Create simple heatmap on right showing action positions
        if i == 0:  # Set up heatmap axes
            # Create a simple grid to show action space
            action_resolution = 10
            x_grid = np.linspace(action_low[0], action_high[0], action_resolution)
            y_grid = np.linspace(action_low[1], action_high[1], action_resolution)
            dummy_grid = np.zeros((action_resolution, action_resolution))
            
            # Plot heatmap with origin='lower' to match mathematical coordinates
            im = ax2.imshow(
                dummy_grid,
                extent=[action_low[0], action_high[0], action_low[1], action_high[1]],
                origin='lower',  # This is the key setting
                cmap='gray',
                aspect='auto',
                alpha=0.3
            )
            ax2.set_title("Action Space Heatmap")
            ax2.set_xlabel("Action X")
            ax2.set_ylabel("Action Y")
            ax2.grid(True, alpha=0.5)
        
        # Plot the test action as a colored cross
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        ax2.plot(action[0], action[1], '+', color=colors[i], markersize=15, 
                markeredgewidth=4, label=f'{label}: ({action[0]:.1f}, {action[1]:.1f})')
    
    ax2.legend()
    
    # Print coordinate system explanation
    print("\n=== Coordinate System Analysis ===")
    print("PushT Action Space:")
    print(f"  - X range: {action_low[0]:.1f} to {action_high[0]:.1f}")
    print(f"  - Y range: {action_low[1]:.1f} to {action_high[1]:.1f}")
    print("\nExpected behavior with origin='lower':")
    print("  - Low Y actions should appear at BOTTOM of heatmap")
    print("  - High Y actions should appear at TOP of heatmap")
    print("  - This should match the environment's visual layout")
    
    plt.tight_layout()
    plt.savefig('coordinate_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Now let's create a more direct test
    print("\n=== Direct Cross Movement Test ===")
    
    # Test just two extreme Y actions to see their visual effect
    low_y_action = np.array([256, action_low[1]])   # Center X, low Y
    high_y_action = np.array([256, action_high[1]])  # Center X, high Y
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test low Y action
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(low_y_action)
    low_y_frame = env.render()
    
    ax1.imshow(low_y_frame)
    ax1.set_title(f'Environment: Action Y={action_low[1]:.1f} (LOW)')
    ax1.axis('off')
    
    # Test high Y action  
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(high_y_action)
    high_y_frame = env.render()
    
    ax2.imshow(high_y_frame)
    ax2.set_title(f'Environment: Action Y={action_high[1]:.1f} (HIGH)')
    ax2.axis('off')
    
    # Show where these should appear on heatmap
    dummy_grid = np.zeros((20, 20))
    
    # Low Y heatmap position
    ax3.imshow(
        dummy_grid,
        extent=[action_low[0], action_high[0], action_low[1], action_high[1]],
        origin='lower',
        cmap='gray',
        alpha=0.3
    )
    ax3.plot(low_y_action[0], low_y_action[1], 'r+', markersize=20, markeredgewidth=4)
    ax3.set_title(f'Heatmap: Low Y Cross at {low_y_action[1]:.1f}')
    ax3.set_xlabel('Action X')
    ax3.set_ylabel('Action Y')
    ax3.grid(True, alpha=0.5)
    
    # High Y heatmap position
    ax4.imshow(
        dummy_grid,
        extent=[action_low[0], action_high[0], action_low[1], action_high[1]],
        origin='lower',
        cmap='gray',
        alpha=0.3
    )
    ax4.plot(high_y_action[0], high_y_action[1], 'r+', markersize=20, markeredgewidth=4)
    ax4.set_title(f'Heatmap: High Y Cross at {high_y_action[1]:.1f}')
    ax4.set_xlabel('Action X')
    ax4.set_ylabel('Action Y')
    ax4.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('cross_movement_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Diagnostic complete! Check the generated images:")
    print("  - coordinate_diagnostic.png")
    print("  - cross_movement_test.png")
    print(f"\nTested actions:")
    print(f"  Low Y action: {low_y_action} (should move agent DOWN/BOTTOM)")
    print(f"  High Y action: {high_y_action} (should move agent UP/TOP)")

if __name__ == "__main__":
    test_coordinate_alignment() 