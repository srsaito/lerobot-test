#!/usr/bin/env python

"""
Comprehensive test to understand PushT coordinate mapping.
"""

import gymnasium as gym
import gym_pusht
import numpy as np
import matplotlib.pyplot as plt

def debug_pusht_coordinates():
    """Systematically test PushT coordinate system."""
    
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=50,
    )
    
    print("Action space bounds:", env.action_space.low, "to", env.action_space.high)
    
    # Test specific action-to-position mappings
    test_actions = [
        [0, 0],      # Bottom-left corner
        [0, 512],    # Top-left corner  
        [512, 0],    # Bottom-right corner
        [512, 512],  # Top-right corner
        [256, 256],  # Center
    ]
    
    results = []
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Test {i+1}: Action {action} ---")
        
        # Reset environment 
        obs, _ = env.reset(seed=42)
        initial_pos = obs["agent_pos"].copy()
        initial_frame = env.render()
        
        # Take multiple steps toward the target to see clear movement
        for step in range(10):
            obs, _, terminated, truncated, _ = env.step(np.array(action, dtype=np.float32))
            if terminated or truncated:
                break
        
        final_pos = obs["agent_pos"].copy()
        final_frame = env.render()
        
        movement = final_pos - initial_pos
        print(f"Action: {action}")
        print(f"Initial pos: {initial_pos}")
        print(f"Final pos: {final_pos}")
        print(f"Movement: {movement}")
        
        # Save comparison images
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(initial_frame)
        axes[0].set_title(f"Initial: Agent at {initial_pos}")
        axes[0].axis('off')
        
        axes[1].imshow(final_frame)
        axes[1].set_title(f"After Action {action}: Agent at {final_pos}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"debug_action_{action[0]}_{action[1]}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        results.append({
            'action': action,
            'initial_pos': initial_pos,
            'final_pos': final_pos,
            'movement': movement
        })
    
    env.close()
    
    # Analyze patterns
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    
    for result in results:
        action = result['action']
        final_pos = result['final_pos']
        print(f"Action {action} → Final pos {final_pos}")
    
    # Look for coordinate system patterns
    print("\nCoordinate System Analysis:")
    print("- Action [0, 0] should correspond to one corner")
    print("- Action [512, 512] should correspond to opposite corner")
    print("- Look at the saved images to see visual correspondence")
    
    # Now test the visualization coordinate system
    print("\n" + "="*50) 
    print("TESTING VISUALIZATION COORDINATES:")
    print("="*50)
    
    # Create a test heatmap to see how imshow displays coordinates
    test_grid = np.zeros((5, 5))
    test_grid[0, 0] = 1  # Top-left in array
    test_grid[4, 4] = 2  # Bottom-right in array
    test_grid[2, 2] = 3  # Center
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Test different imshow configurations
    axes[0].imshow(test_grid, origin='lower', extent=[0, 512, 0, 512])
    axes[0].set_title("origin='lower' (Mathematical)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    axes[1].imshow(test_grid, origin='upper', extent=[0, 512, 0, 512])
    axes[1].set_title("origin='upper' (Screen)")
    axes[1].set_xlabel("X") 
    axes[1].set_ylabel("Y")
    
    axes[2].imshow(np.flipud(test_grid), origin='lower', extent=[0, 512, 0, 512])
    axes[2].set_title("flipud + origin='lower'")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig("coordinate_test_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created visualization coordinate test: coordinate_test_visualization.png")
    print("\nIn this test:")
    print("- Value 1 (red) should be at action coordinate (0,0)")
    print("- Value 2 (yellow) should be at action coordinate (512,512)")
    print("- Value 3 (green) should be at action coordinate (256,256)")

if __name__ == "__main__":
    debug_pusht_coordinates() 