#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualize ACT Policy Action Landscape for PushT Environment

This script creates a side-by-side visualization:
- Left: Expert demonstration replay in PushT environment
- Right: Action space heatmap showing policy scores for all possible actions at each timestep

Usage:
python visualize_act_policy_landscape.py \
    --policy_path lerobot/act_pusht \
    --dataset_repo_id lerobot/pusht \
    --episode_idx 0 \
    --output_dir outputs/act_landscape_viz \
    --action_resolution 50 \
    --max_steps 100
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import init_logging


class ACTPolicyLandscapeVisualizer:
    """Visualizes ACT policy action landscape alongside expert demonstrations."""
    
    def __init__(
        self,
        policy_path: str,
        dataset_repo_id: str,
        device: str = "cuda",
        action_resolution: int = 50,
    ):
        """
        Args:
            policy_path: Path to trained ACT policy
            dataset_repo_id: HuggingFace dataset repository ID
            device: Device to run policy on
            action_resolution: Resolution for action space sampling (NxN grid)
        """
        self.device = device
        self.action_resolution = action_resolution
        
        # Load policy
        logging.info(f"Loading ACT policy from {policy_path}")
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.to(device)
        self.policy.eval()
        
        # Load dataset
        logging.info(f"Loading dataset {dataset_repo_id}")
        self.dataset = LeRobotDataset(dataset_repo_id)
        
        # Create environment
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=300,
        )
        
        # Get action space bounds
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        logging.info(f"Action space: low={self.action_low}, high={self.action_high}")
        
        # Create action grid for sampling
        self.action_grid = self._create_action_grid()
        
    def _create_action_grid(self) -> torch.Tensor:
        """Create a grid of actions to sample across the action space."""
        # Create meshgrid for 2D action space (x, y)
        x_actions = np.linspace(self.action_low[0], self.action_high[0], self.action_resolution)
        y_actions = np.linspace(self.action_low[1], self.action_high[1], self.action_resolution)
        
        xx, yy = np.meshgrid(x_actions, y_actions)
        actions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return torch.from_numpy(actions).float().to(self.device)
    
    def _prepare_observation_for_policy(self, obs_dict: Dict) -> Dict[str, torch.Tensor]:
        """Convert environment observation to policy input format."""
        # Convert numpy arrays to tensors and normalize
        state = torch.from_numpy(obs_dict["agent_pos"]).float()
        image = torch.from_numpy(obs_dict["pixels"]).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW
        
        # Add batch dimension and move to device
        state = state.unsqueeze(0).to(self.device)
        image = image.unsqueeze(0).to(self.device)
        
        return {
            "observation.state": state,
            "observation.image": image,
        }
    
    def _get_policy_scores_for_actions(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get policy scores for a batch of actions given an observation.
        
        Args:
            observation: Policy observation dict
            actions: Tensor of shape (N, action_dim) containing actions to evaluate
            
        Returns:
            Tensor of shape (N,) containing policy scores for each action
        """
        batch_size = actions.shape[0]
        
        # Expand observation to match action batch size
        expanded_obs = {}
        for key, value in observation.items():
            expanded_obs[key] = value.expand(batch_size, -1, -1, -1) if "image" in key else value.expand(batch_size, -1)
        
        # Prepare action tensor with proper shape for policy
        # ACT expects actions with chunk_size dimension
        chunk_size = self.policy.config.chunk_size
        expanded_actions = actions.unsqueeze(1).expand(-1, chunk_size, -1)  # (batch, chunk_size, action_dim)
        
        # Create action padding mask (no padding for this evaluation)
        action_is_pad = torch.zeros(batch_size, chunk_size, dtype=torch.bool, device=self.device)
        
        # Prepare batch for policy
        batch = {
            **expanded_obs,
            "action": expanded_actions,
            "action_is_pad": action_is_pad,
        }
        
        # Get policy predictions and compute negative loss as score
        # (higher score = better action according to policy)
        with torch.no_grad():
            loss, loss_dict = self.policy.forward(batch)
            # Use negative L1 loss as score (lower loss = higher score)
            scores = -loss_dict["l1_loss"] * torch.ones(batch_size, device=self.device)
        
        return scores
    
    def _create_action_heatmap(
        self, 
        observation: Dict[str, torch.Tensor], 
        expert_action: np.ndarray
    ) -> np.ndarray:
        """Create action space heatmap for given observation."""
        # Get policy scores for all actions in the grid
        scores = self._get_policy_scores_for_actions(observation, self.action_grid)
        
        # Reshape scores to 2D grid
        score_grid = scores.cpu().numpy().reshape(self.action_resolution, self.action_resolution)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create heatmap with colormap
        # Use inverted coordinates (origin='upper') to match PushT's coordinate system
        # Low Y action values → agent at TOP of environment → TOP of heatmap
        im = ax.imshow(
            score_grid,  # Use raw score grid
            extent=[self.action_low[0], self.action_high[0], self.action_low[1], self.action_high[1]],
            origin='upper',  # Inverted coordinates: low Y at top (matches PushT's visual layout)
            cmap='viridis',
            aspect='auto'
        )
        
        # Mark expert action with red cross
        # When using origin='upper', we need to flip Y coordinates for plotting
        plot_expert_action = expert_action.copy()
        plot_expert_action[1] = self.action_high[1] - expert_action[1] + self.action_low[1]
        
        ax.plot(plot_expert_action[0], plot_expert_action[1], 'r+', markersize=15, markeredgewidth=3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Policy Score', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Action X')
        ax.set_ylabel('Action Y')
        ax.set_title('ACT Policy Action Landscape')
        
        # Convert plot to image array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def visualize_episode(
        self, 
        episode_idx: int, 
        output_dir: Path, 
        max_steps: int = None
    ) -> None:
        """
        Create side-by-side visualization for an episode.
        
        Args:
            episode_idx: Episode index from dataset
            output_dir: Directory to save outputs
            max_steps: Maximum number of steps to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get episode data from dataset
        episode = self.dataset.get_episode(episode_idx)
        expert_actions = episode["action"].numpy()
        
        if max_steps is not None:
            expert_actions = expert_actions[:max_steps]
        
        # Reset environment with same seed as dataset episode
        seed = episode["seed"][0].item()
        obs, _ = self.env.reset(seed=seed)
        
        # Storage for frames
        left_frames = []  # Environment frames
        right_frames = []  # Action landscape heatmaps
        combined_frames = []
        
        logging.info(f"Generating visualization for episode {episode_idx} with {len(expert_actions)} steps")
        
        for step in tqdm(range(len(expert_actions))):
            # Get current expert action
            expert_action = expert_actions[step]
            
            # Render current environment state
            env_frame = self.env.render()
            left_frames.append(env_frame)
            
            # Prepare observation for policy
            policy_obs = self._prepare_observation_for_policy(obs)
            
            # Create action landscape heatmap
            heatmap_frame = self._create_action_heatmap(policy_obs, expert_action)
            right_frames.append(heatmap_frame)
            
            # Combine frames side by side
            # Resize frames to same height if needed
            left_height, left_width = env_frame.shape[:2]
            right_height, right_width = heatmap_frame.shape[:2]
            
            if left_height != right_height:
                # Resize to match heights
                target_height = min(left_height, right_height)
                left_ratio = target_height / left_height
                right_ratio = target_height / right_height
                
                left_new_width = int(left_width * left_ratio)
                right_new_width = int(right_width * right_ratio)
                
                # Use simple resizing (you could use cv2 for better quality)
                env_frame_resized = np.array(env_frame)  # Keep original for now
                heatmap_frame_resized = np.array(heatmap_frame)  # Keep original for now
            else:
                env_frame_resized = env_frame
                heatmap_frame_resized = heatmap_frame
            
            # Combine horizontally
            combined_frame = np.hstack([env_frame_resized, heatmap_frame_resized])
            combined_frames.append(combined_frame)
            
            # Step environment with expert action
            obs, _, terminated, truncated, _ = self.env.step(expert_action)
            
            if terminated or truncated:
                break
        
        # Save individual frame sequences
        logging.info("Saving frame sequences...")
        
        # Save environment frames
        env_video_path = output_dir / f"episode_{episode_idx}_environment.mp4"
        imageio.mimsave(str(env_video_path), left_frames, fps=10)
        
        # Save heatmap frames
        heatmap_video_path = output_dir / f"episode_{episode_idx}_heatmaps.mp4"
        imageio.mimsave(str(heatmap_video_path), right_frames, fps=10)
        
        # Save combined video
        combined_video_path = output_dir / f"episode_{episode_idx}_combined.mp4"
        imageio.mimsave(str(combined_video_path), combined_frames, fps=10)
        
        logging.info(f"Videos saved:")
        logging.info(f"  Environment: {env_video_path}")
        logging.info(f"  Heatmaps: {heatmap_video_path}")
        logging.info(f"  Combined: {combined_video_path}")
        
        # Save summary statistics
        stats = {
            "episode_idx": episode_idx,
            "num_steps": len(expert_actions),
            "seed": seed,
            "action_resolution": self.action_resolution,
        }
        
        import json
        with open(output_dir / f"episode_{episode_idx}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT Policy Action Landscape")
    parser.add_argument("--policy_path", type=str, required=True, 
                       help="Path to trained ACT policy")
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/pusht",
                       help="HuggingFace dataset repository ID")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to visualize")
    parser.add_argument("--output_dir", type=str, default="outputs/act_landscape_viz",
                       help="Output directory for videos")
    parser.add_argument("--action_resolution", type=int, default=50,
                       help="Resolution for action space sampling (NxN grid)")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of steps to visualize")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run policy on")
    
    args = parser.parse_args()
    
    init_logging()
    
    # Create visualizer
    visualizer = ACTPolicyLandscapeVisualizer(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        device=args.device,
        action_resolution=args.action_resolution,
    )
    
    # Generate visualization
    visualizer.visualize_episode(
        episode_idx=args.episode_idx,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    
    logging.info("Visualization complete!")


if __name__ == "__main__":
    main() 