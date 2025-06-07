#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_pusht

def test_action_to_position_mapping():
    """Test how PushT actions map to agent positions in the environment."""
    
    print("=== Testing Action-to-Position Mapping ===")
    
    # Create environment
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )
    
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"Action space: {action_low} to {action_high}")
    
    # Test extreme Y actions to see where agent ends up
    test_actions = [
        np.array([256, 50]),    # Low Y action
        np.array([256, 256]),   # Middle Y action  
        np.array([256, 450]),   # High Y action
    ]
    
    labels = ["Low Y (50)", "Middle Y (256)", "High Y (450)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (action, label) in enumerate(zip(test_actions, labels)):
        print(f"\nTesting {label}: action = {action}")
        
        # Reset environment
        obs, _ = env.reset()
        initial_agent_pos = obs["agent_pos"].copy()
        print(f"  Initial agent_pos: {initial_agent_pos}")
        
        # Take action
        obs, _, _, _, _ = env.step(action)
        final_agent_pos = obs["agent_pos"].copy()
        print(f"  Final agent_pos: {final_agent_pos}")
        
        # Render environment
        env_frame = env.render()
        
        # Show environment
        axes[i].imshow(env_frame)
        axes[i].set_title(f'{label}\nAction: {action}\nAgent pos: {final_agent_pos}')
        axes[i].axis('off')
        
        # Calculate movement
        movement = final_agent_pos - initial_agent_pos
        print(f"  Movement: {movement}")
        
        # Check if Y movement matches Y action expectation
        y_action_value = action[1]
        y_movement = movement[1]
        
        print(f"  Y action: {y_action_value} (higher values should move agent UP)")
        print(f"  Y movement: {y_movement} (positive = moved up, negative = moved down)")
        
        if y_action_value < 256:  # Low Y action
            expected = "agent should move DOWN (negative Y movement)"
        else:  # High Y action
            expected = "agent should move UP (positive Y movement)"
        print(f"  Expected: {expected}")
        
        # Check if the visual position matches
        # In image coordinates, (0,0) is top-left, so higher pixel Y = lower visual position
        frame_height = env_frame.shape[0]
        agent_pixel_y = final_agent_pos[1] if hasattr(final_agent_pos, '__getitem__') else final_agent_pos
        
        visual_position = "UNKNOWN"
        try:
            if isinstance(final_agent_pos, np.ndarray) and len(final_agent_pos) >= 2:
                # Agent pos is in environment coordinates, need to see where it appears visually
                visual_position = f"env_coords({final_agent_pos[0]:.1f}, {final_agent_pos[1]:.1f})"
        except:
            pass
            
        print(f"  Visual position: {visual_position}")
    
    plt.tight_layout()
    plt.savefig('action_to_position_mapping.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Analysis ===")
    print("Key insight: We need to understand the relationship between:")
    print("1. Action Y values (0-512)")
    print("2. Agent position coordinates") 
    print("3. Visual position in rendered image")
    print("\nIf low Y actions move the agent to visually HIGH positions,")
    print("then our heatmap coordinate system might need adjustment.")

if __name__ == "__main__":
    test_action_to_position_mapping() 