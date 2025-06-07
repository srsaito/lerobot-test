#!/usr/bin/env python

"""
Advanced ACT Policy Action Landscape Visualization

This enhanced version includes:
- Multiple scoring methods (L1 loss, confidence, likelihood)
- Policy prediction arrows showing chosen action
- Confidence intervals visualization
- Batch processing for efficiency
- Customizable colormaps and visualization options

Usage:
python visualize_act_landscape_advanced.py \
    --policy_path lerobot/act_pusht \
    --dataset_repo_id lerobot/pusht \
    --episode_idx 0 \
    --output_dir outputs/act_landscape_viz_advanced \
    --action_resolution 50 \
    --scoring_method confidence \
    --show_policy_prediction true \
    --colormap plasma
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Circle
from tqdm import tqdm
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import init_logging


class AdvancedACTPolicyLandscapeVisualizer:
    """Advanced visualizer for ACT policy action landscapes with multiple scoring methods."""
    
    def __init__(
        self,
        policy_path: str,
        dataset_repo_id: str,
        device: str = "cuda",
        action_resolution: int = 50,
        scoring_method: str = "confidence",
        colormap: str = "viridis",
        batch_size: int = 100,
    ):
        """
        Args:
            policy_path: Path to trained ACT policy
            dataset_repo_id: HuggingFace dataset repository ID
            device: Device to run policy on
            action_resolution: Resolution for action space sampling (NxN grid)
            scoring_method: Method for scoring actions ('l1_loss', 'confidence', 'likelihood')
            colormap: Matplotlib colormap for heatmap
            batch_size: Batch size for efficient action evaluation
        """
        self.device = device
        self.action_resolution = action_resolution
        self.scoring_method = scoring_method
        self.colormap = colormap
        self.batch_size = batch_size
        
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
        x_actions = np.linspace(self.action_low[0], self.action_high[0], self.action_resolution)
        y_actions = np.linspace(self.action_low[1], self.action_high[1], self.action_resolution)
        
        xx, yy = np.meshgrid(x_actions, y_actions)
        actions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return torch.from_numpy(actions).float().to(self.device)
    
    def _prepare_observation_for_policy(self, obs_dict: Dict) -> Dict[str, torch.Tensor]:
        """Convert environment observation to policy input format."""
        state = torch.from_numpy(obs_dict["agent_pos"]).float()
        image = torch.from_numpy(obs_dict["pixels"]).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW
        
        state = state.unsqueeze(0).to(self.device)
        image = image.unsqueeze(0).to(self.device)
        
        return {
            "observation.state": state,
            "observation.image": image,
        }
    
    def _get_policy_prediction(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get the policy's predicted action for the current observation."""
        with torch.no_grad():
            action = self.policy.select_action(observation)
            return action.cpu().numpy().flatten()
    
    def _get_policy_scores_batch(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get policy scores for a batch of actions with different scoring methods.
        
        Args:
            observation: Policy observation dict
            actions: Tensor of shape (N, action_dim) containing actions to evaluate
            
        Returns:
            Tensor of shape (N,) containing policy scores for each action
        """
        all_scores = []
        
        # Process actions in batches for memory efficiency
        for i in range(0, len(actions), self.batch_size):
            batch_actions = actions[i:i+self.batch_size]
            batch_size = batch_actions.shape[0]
            
            # Expand observation to match action batch size
            expanded_obs = {}
            for key, value in observation.items():
                if "image" in key:
                    expanded_obs[key] = value.expand(batch_size, -1, -1, -1)
                else:
                    expanded_obs[key] = value.expand(batch_size, -1)
            
            # Prepare action tensor with proper shape for policy
            chunk_size = self.policy.config.chunk_size
            expanded_actions = batch_actions.unsqueeze(1).expand(-1, chunk_size, -1)
            
            # Create action padding mask
            action_is_pad = torch.zeros(batch_size, chunk_size, dtype=torch.bool, device=self.device)
            
            # Prepare batch for policy
            batch = {
                **expanded_obs,
                "action": expanded_actions,
                "action_is_pad": action_is_pad,
            }
            
            with torch.no_grad():
                if self.scoring_method == "l1_loss":
                    # Use negative L1 loss as score
                    loss, loss_dict = self.policy.forward(batch)
                    scores = -torch.full((batch_size,), loss_dict["l1_loss"], device=self.device)
                    
                elif self.scoring_method == "confidence":
                    # Use model confidence based on action prediction consistency
                    predicted_actions, _ = self.policy.model(batch)
                    
                    # Calculate confidence as negative variance across chunk
                    action_var = torch.var(predicted_actions, dim=1).mean(dim=1)
                    scores = -action_var
                    
                elif self.scoring_method == "likelihood":
                    # Use action likelihood under the policy
                    predicted_actions, (mu, log_sigma_x2) = self.policy.model(batch)
                    
                    if mu is not None and log_sigma_x2 is not None:
                        # Calculate log likelihood under Gaussian distribution
                        diff = batch_actions.unsqueeze(1) - predicted_actions
                        sigma_x2 = torch.exp(log_sigma_x2)
                        log_likelihood = -0.5 * (diff**2 / sigma_x2 + log_sigma_x2 + np.log(2 * np.pi))
                        scores = log_likelihood.mean(dim=[1, 2])  # Average over chunk and action dims
                    else:
                        # Fallback to L2 distance if no VAE
                        diff = batch_actions.unsqueeze(1) - predicted_actions
                        scores = -torch.norm(diff, dim=2).mean(dim=1)
                
                else:
                    raise ValueError(f"Unknown scoring method: {self.scoring_method}")
            
            all_scores.append(scores)
        
        return torch.cat(all_scores, dim=0)
    
    def _create_enhanced_action_heatmap(
        self, 
        observation: Dict[str, torch.Tensor], 
        expert_action: np.ndarray,
        show_policy_prediction: bool = True,
        show_confidence_intervals: bool = False,
    ) -> np.ndarray:
        """Create enhanced action space heatmap with additional visualizations."""
        # Get policy scores for all actions in the grid
        scores = self._get_policy_scores_batch(observation, self.action_grid)
        
        # Reshape scores to 2D grid
        score_grid = scores.cpu().numpy().reshape(self.action_resolution, self.action_resolution)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create heatmap with custom colormap
        # Use inverted coordinates (origin='upper') to match PushT's coordinate system
        # In PushT: Low Y action values → agent appears at TOP of environment → TOP of heatmap
        # In PushT: High Y action values → agent appears at BOTTOM of environment → BOTTOM of heatmap
        im = ax.imshow(
            score_grid,  # Use raw score grid
            extent=[self.action_low[0], self.action_high[0], self.action_low[1], self.action_high[1]],
            origin='upper',  # Inverted coordinates: low Y at top (matches PushT's visual layout)
            cmap=self.colormap,
            aspect='auto',
            alpha=0.8
        )
        
        # Mark expert action with red cross
        # When using origin='upper', we need to flip Y coordinates for plotting
        plot_expert_action = expert_action.copy()
        plot_expert_action[1] = self.action_high[1] - expert_action[1] + self.action_low[1]
        
        ax.plot(plot_expert_action[0], plot_expert_action[1], 'r+', markersize=20, markeredgewidth=4, 
                label='Expert Action')
        
        # Show policy prediction if requested
        if show_policy_prediction:
            policy_action = self._get_policy_prediction(observation)
            plot_policy_action = policy_action.copy()
            plot_policy_action[1] = self.action_high[1] - policy_action[1] + self.action_low[1]
            
            ax.plot(plot_policy_action[0], plot_policy_action[1], 'w*', markersize=15, markeredgewidth=2, 
                    markeredgecolor='black', label='Policy Prediction')
            
            # Draw arrow from policy prediction to expert action
            arrow = FancyArrowPatch(
                (plot_policy_action[0], plot_policy_action[1]),
                (plot_expert_action[0], plot_expert_action[1]),
                arrowstyle='->',
                mutation_scale=20,
                color='yellow',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(arrow)
        
        # Add confidence intervals if requested
        if show_confidence_intervals:
            # Find high-confidence regions (top 10% of scores)
            threshold = np.percentile(score_grid, 90)
            high_conf_mask = score_grid >= threshold
            
            # Add contour for high-confidence regions
            ax.contour(
                np.linspace(self.action_low[0], self.action_high[0], self.action_resolution),
                np.linspace(self.action_low[1], self.action_high[1], self.action_resolution),
                high_conf_mask.astype(float),
                levels=[0.5],
                colors='white',
                linewidths=2,
                alpha=0.8
            )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'Policy Score ({self.scoring_method})', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Action X', fontsize=12)
        ax.set_ylabel('Action Y', fontsize=12)
        ax.set_title(f'ACT Policy Action Landscape\n({self.scoring_method.replace("_", " ").title()})', 
                    fontsize=14)
        ax.legend(loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array
    
    def _resize_frame(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        """Resize frame to target height while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))
    
    def visualize_episode(
        self, 
        episode_idx: int, 
        output_dir: Path, 
        max_steps: Optional[int] = None,
        show_policy_prediction: bool = True,
        show_confidence_intervals: bool = False,
        target_height: int = 480,
    ) -> None:
        """
        Create advanced side-by-side visualization for an episode.
        
        Args:
            episode_idx: Episode index from dataset
            output_dir: Directory to save outputs
            max_steps: Maximum number of steps to visualize
            show_policy_prediction: Whether to show policy's predicted action
            show_confidence_intervals: Whether to show confidence interval contours
            target_height: Target height for video frames
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
        
        logging.info(f"Generating advanced visualization for episode {episode_idx} with {len(expert_actions)} steps")
        logging.info(f"Scoring method: {self.scoring_method}, Colormap: {self.colormap}")
        
        for step in tqdm(range(len(expert_actions))):
            # Get current expert action
            expert_action = expert_actions[step]
            
            # Render current environment state
            env_frame = self.env.render()
            left_frames.append(env_frame)
            
            # Prepare observation for policy
            policy_obs = self._prepare_observation_for_policy(obs)
            
            # Create enhanced action landscape heatmap
            heatmap_frame = self._create_enhanced_action_heatmap(
                policy_obs, 
                expert_action,
                show_policy_prediction=show_policy_prediction,
                show_confidence_intervals=show_confidence_intervals,
            )
            right_frames.append(heatmap_frame)
            
            # Resize frames to target height
            env_frame_resized = self._resize_frame(env_frame, target_height)
            heatmap_frame_resized = self._resize_frame(heatmap_frame, target_height)
            
            # Combine horizontally
            combined_frame = np.hstack([env_frame_resized, heatmap_frame_resized])
            combined_frames.append(combined_frame)
            
            # Step environment with expert action
            obs, _, terminated, truncated, _ = self.env.step(expert_action)
            
            if terminated or truncated:
                break
        
        # Save videos
        logging.info("Saving frame sequences...")
        
        fps = 10
        env_video_path = output_dir / f"episode_{episode_idx}_environment.mp4"
        imageio.mimsave(str(env_video_path), left_frames, fps=fps)
        
        heatmap_video_path = output_dir / f"episode_{episode_idx}_heatmaps_{self.scoring_method}.mp4"
        imageio.mimsave(str(heatmap_video_path), right_frames, fps=fps)
        
        combined_video_path = output_dir / f"episode_{episode_idx}_combined_{self.scoring_method}.mp4"
        imageio.mimsave(str(combined_video_path), combined_frames, fps=fps)
        
        logging.info(f"Videos saved:")
        logging.info(f"  Environment: {env_video_path}")
        logging.info(f"  Heatmaps: {heatmap_video_path}")
        logging.info(f"  Combined: {combined_video_path}")
        
        # Save configuration and statistics
        config = {
            "episode_idx": episode_idx,
            "num_steps": len(expert_actions),
            "seed": seed,
            "action_resolution": self.action_resolution,
            "scoring_method": self.scoring_method,
            "colormap": self.colormap,
            "show_policy_prediction": show_policy_prediction,
            "show_confidence_intervals": show_confidence_intervals,
            "target_height": target_height,
        }
        
        import json
        with open(output_dir / f"episode_{episode_idx}_config.json", "w") as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Advanced ACT Policy Action Landscape Visualization")
    parser.add_argument("--policy_path", type=str, required=True, 
                       help="Path to trained ACT policy")
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/pusht",
                       help="HuggingFace dataset repository ID")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to visualize")
    parser.add_argument("--output_dir", type=str, default="outputs/act_landscape_viz_advanced",
                       help="Output directory for videos")
    parser.add_argument("--action_resolution", type=int, default=50,
                       help="Resolution for action space sampling (NxN grid)")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of steps to visualize")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run policy on")
    parser.add_argument("--scoring_method", type=str, default="confidence",
                       choices=["l1_loss", "confidence", "likelihood"],
                       help="Method for scoring actions")
    parser.add_argument("--colormap", type=str, default="viridis",
                       help="Matplotlib colormap for heatmap")
    parser.add_argument("--show_policy_prediction", type=bool, default=True,
                       help="Whether to show policy's predicted action")
    parser.add_argument("--show_confidence_intervals", type=bool, default=False,
                       help="Whether to show confidence interval contours")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for efficient action evaluation")
    parser.add_argument("--target_height", type=int, default=480,
                       help="Target height for video frames")
    
    args = parser.parse_args()
    
    init_logging()
    
    # Create visualizer
    visualizer = AdvancedACTPolicyLandscapeVisualizer(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        device=args.device,
        action_resolution=args.action_resolution,
        scoring_method=args.scoring_method,
        colormap=args.colormap,
        batch_size=args.batch_size,
    )
    
    # Generate visualization
    visualizer.visualize_episode(
        episode_idx=args.episode_idx,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        show_policy_prediction=args.show_policy_prediction,
        show_confidence_intervals=args.show_confidence_intervals,
        target_height=args.target_height,
    )
    
    logging.info("Advanced visualization complete!")


if __name__ == "__main__":
    main() 