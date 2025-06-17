#!/usr/bin/env python3

"""
Multi-Policy Action Landscape Visualizer

Supports visualization of action landscapes for different policy types:
- Diffusion Policy
- ACT (Action Chunking Transformer)
- TD-MPC (future)
- VQ-BeT (future)

Usage:
    # For Diffusion Policy
    python visualize_policy_landscape_multi.py \
        --policy_type diffusion \
        --policy_path lerobot/diffusion_pusht \
        --episode_idx 0 \
        --max_steps 30

    # For ACT Policy (placeholder)
    python visualize_policy_landscape_multi.py \
        --policy_type act \
        --policy_path PLACEHOLDER_ACT_MODEL \
        --episode_idx 0 \
        --max_steps 10
"""

import argparse
import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
# from matplotlib.patches import FancyArrowPatch  # Removed - no longer using arrows
from matplotlib.patches import Circle
from tqdm import tqdm
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import init_logging


class BasePolicyLandscapeVisualizer(ABC):
    """Base class for policy landscape visualization with comprehensive fallback handling."""
    
    def __init__(
        self,
        policy_path: str,
        dataset_repo_id: str,
        device: str = "cuda",
        action_resolution: int = 50,
        scoring_method: str = "likelihood",
        colormap: str = "viridis",
        batch_size: int = 100,
        trace_length: int = 10,
    ):
        self.policy_path = policy_path
        self.dataset_repo_id = dataset_repo_id
        self.device = device
        self.action_resolution = action_resolution
        self.scoring_method = scoring_method
        self.effective_scoring_method = scoring_method  # May change due to persistent failures
        self.colormap = colormap
        self.batch_size = batch_size
        self.trace_length = trace_length
        
        # Fallback tracking and statistics
        self.fallback_stats = {
            'kde_failures': 0,
            'memory_errors': 0,
            'device_errors': 0,
            'policy_errors': 0,
            'numerical_errors': 0,
            'total_steps': 0,
            'successful_steps': 0,
            'frozen_steps': 0  # Steps where we used last good state
        }
        self.scoring_failures = 0
        self.last_good_scores = None  # Cache for "freeze last good state" fallback
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Switch to distance after this many
        
        # Initialize environment and policy
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=300,
        )
        # Get action space bounds from environment
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        
        # Load dataset
        self.dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
        
        # Create action grid for evaluation
        self.action_grid = self._create_action_grid()
        
        # Load the policy
        self._load_policy()
        
        # Initialize action trace
        self.action_trace = []
        
        # Log initialization info
        logging.info(f"Action space: low={self.action_low}, high={self.action_high}")
        logging.info(f"Initialized visualizer with device: {self.device}")
        logging.info(f"Action space: [{self.action_low[0]}, {self.action_high[0]}] x [{self.action_low[1]}, {self.action_high[1]}]")
        logging.info(f"Action grid shape: {self.action_grid.shape}")
        logging.info(f"Scoring method: {self.scoring_method}")
        logging.info(f"Batch sizes: main={self.batch_size}, sampling=50 (adjustable)")
        
    @abstractmethod
    def _load_policy(self):
        """Load the policy model."""
        pass
    
    @abstractmethod
    def _get_policy_name(self) -> str:
        """Get the policy name for display."""
        pass
        
    def _create_action_grid(self) -> torch.Tensor:
        """Create a grid of actions to sample across the action space."""
        x_actions = np.linspace(self.action_low[0], self.action_high[0], self.action_resolution)
        y_actions = np.linspace(self.action_low[1], self.action_high[1], self.action_resolution)
        
        xx, yy = np.meshgrid(x_actions, y_actions)
        actions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return torch.from_numpy(actions).float().to(self.device)
    
    def _prepare_observation_for_policy(self, obs_dict: Dict, dataset_image: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Convert environment observation to policy input format."""
        state = torch.from_numpy(obs_dict["agent_pos"]).float()
        
        # Use dataset image if provided, otherwise use environment's rendered image
        if dataset_image is not None:
            # Dataset images are already in CHW format and [0, 1] range
            image = dataset_image.float()
        else:
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
    
    def _update_action_trace(self, expert_action: np.ndarray) -> None:
        """Update the trace of recent expert actions."""
        self.action_trace.append(expert_action.copy())
        if len(self.action_trace) > self.trace_length:
            self.action_trace.pop(0)

    def _update_policy_trace(self, policy_action: np.ndarray) -> None:
        """Update the trace of recent policy predictions."""
        if not hasattr(self, 'policy_trace'):
            self.policy_trace = []
        self.policy_trace.append(policy_action.copy())
        if len(self.policy_trace) > self.trace_length:
            self.policy_trace.pop(0)

    def _reset_action_trace(self) -> None:
        """Reset the action trace for a new episode."""
        self.action_trace = []

    def _reset_policy_trace(self) -> None:
        """Reset the policy trace for a new episode."""
        self.policy_trace = []
    
    def _load_episode_initial_state(self, episode_idx: int) -> np.ndarray:
        """Load the complete initial state for an episode from the raw zarr dataset.
        
        Returns:
            Initial state [agent_x, agent_y, block_x, block_y, block_angle]
        """
        try:
            from zarr_dataset_manager import get_zarr_dataset_path
            import zarr
            
            # Get path to zarr dataset using our manager
            dataset_path = get_zarr_dataset_path()
            state_group = zarr.open(dataset_path / "data" / "state", mode='r')
            initial_state = state_group[episode_idx]  # Get initial state for this episode
            return np.array(initial_state)
        except Exception as e:
            logging.warning(f"Could not load full initial state from zarr: {e}")
            logging.warning("Falling back to agent position only from LeRobot dataset")
            # Fallback to LeRobot dataset (agent position only)
            from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
            episode_data = self.dataset.hf_dataset.select(range(from_idx, from_idx + 1))
            agent_pos = episode_data["observation.state"][0].numpy()
            # Create dummy state with random T-shape position
            return np.array([agent_pos[0], agent_pos[1], 256.0, 256.0, 0.0])
    
    def _load_episode_image(self, episode_idx: int, step_idx: int) -> torch.Tensor:
        """Load the image for a specific step from the dataset.
        
        Args:
            episode_idx: Episode index
            step_idx: Step index within the episode
            
        Returns:
            Image tensor of shape (3, 96, 96) with values in [0, 1]
        """
        try:
            # Calculate global step index
            from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
            global_step_idx = from_idx + step_idx
            
            # Load image directly from dataset (already decoded from video)
            image = self.dataset[global_step_idx]["observation.image"]
            
            return image
            
        except Exception as e:
            logging.warning(f"Could not load image from dataset for episode {episode_idx}, step {step_idx}: {e}")
            logging.warning("Falling back to environment rendered image")
            return None
    
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
            
            with torch.no_grad():
                if self.effective_scoring_method == "likelihood":
                    # Use the diffusion model's likelihood of generating these actions
                    scores = self._compute_action_likelihood(expanded_obs, batch_actions)
                    
                elif self.effective_scoring_method == "distance":
                    # Use negative distance to policy's predicted action
                    policy_action = self._get_policy_prediction(observation)
                    policy_action_tensor = torch.from_numpy(policy_action).float().to(self.device)
                    distances = torch.norm(batch_actions - policy_action_tensor, dim=1)
                    scores = -distances  # Negative distance (closer = higher score)
                    
                elif self.effective_scoring_method == "noise_pred":
                    # Use the noise prediction quality as a score
                    scores = self._compute_noise_prediction_score(expanded_obs, batch_actions)
                
                else:
                    raise ValueError(f"Unknown scoring method: {self.effective_scoring_method}")
            
            all_scores.append(scores)
        
        return torch.cat(all_scores, dim=0)
    
    @abstractmethod
    def _compute_action_likelihood(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute likelihood of actions under the diffusion policy."""
        batch_size = actions.shape[0]
        
        # Prepare actions with temporal dimension for diffusion model
        horizon = self.policy.config.horizon
        n_action_steps = self.policy.config.n_action_steps
        
        # Expand actions to match horizon (repeat the action across time)
        expanded_actions = actions.unsqueeze(1).expand(-1, horizon, -1)  # (batch, horizon, action_dim)
        
        # Create action padding mask (no padding for this evaluation)
        action_is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=self.device)
        
        # Prepare batch for policy
        batch = {
            **observation,
            "action": expanded_actions,
            "action_is_pad": action_is_pad,
        }
        
        # Use the diffusion model to get likelihood
        try:
            # For diffusion policies, we can compute the negative MSE loss as a likelihood proxy
            predicted_actions = self.policy.diffusion.generate_actions(batch)
            
            # Compute negative MSE between target and predicted actions
            mse = F.mse_loss(
                predicted_actions[:, :n_action_steps], 
                expanded_actions[:, :n_action_steps], 
                reduction='none'
            ).mean(dim=[1, 2])
            
            scores = -mse  # Higher score for better predictions
            
        except Exception as e:
            return self._handle_fallback(e, "likelihood_computation", actions)
        
        # Record successful computation
        self._record_successful_computation(scores)
        return scores
    
    def _compute_noise_prediction_score(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute score based on noise prediction quality."""
        batch_size = actions.shape[0]
        horizon = self.policy.config.horizon
        
        # Expand actions to match horizon
        expanded_actions = actions.unsqueeze(1).expand(-1, horizon, -1)
        
        try:
            # Add noise to actions and see how well the model can predict it
            noise_level = 0.1
            noise = torch.randn_like(expanded_actions) * noise_level
            noisy_actions = expanded_actions + noise
            
            # Use the diffusion model's denoising capability as a score
            action_is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=self.device)
            
            batch = {
                **observation,
                "action": noisy_actions,
                "action_is_pad": action_is_pad,
            }
            
            # Get denoised prediction
            denoised_actions = self.policy.diffusion.generate_actions(batch)
            
            # Score based on how well it denoised back to original
            denoising_error = F.mse_loss(
                denoised_actions, 
                expanded_actions, 
                reduction='none'
            ).mean(dim=[1, 2])
            
            scores = -denoising_error  # Lower error = higher score
            
        except Exception as e:
            return self._handle_fallback(e, "noise_prediction", actions)
        
        # Record successful computation
        self._record_successful_computation(scores)
        return scores
    
    def _create_enhanced_action_heatmap(
        self, 
        observation: Dict[str, torch.Tensor], 
        action: np.ndarray,
        expert_mode: bool,
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
        
        # In expert mode, show expert action trace and current expert action
        if expert_mode:
            # Draw expert action trace/snake if we have history
            if len(self.action_trace) > 1:
                # Transform all trace actions for plotting
                trace_points = []
                for trace_action in self.action_trace:
                    plot_trace_action = trace_action.copy()
                    plot_trace_action[1] = self.action_high[1] - trace_action[1] + self.action_low[1]
                    trace_points.append(plot_trace_action)
                
                trace_points = np.array(trace_points)
                
                # Draw the trace line with varying alpha (newer points more opaque)
                for i in range(len(trace_points) - 1):
                    alpha = (i + 1) / len(trace_points)  # Increase alpha for newer segments
                    ax.plot(
                        trace_points[i:i+2, 0], 
                        trace_points[i:i+2, 1], 
                        'r-', 
                        linewidth=3, 
                        alpha=alpha * 0.8,
                        solid_capstyle='round'
                    )
                
                # Draw small circles for past positions
                for i, point in enumerate(trace_points[:-1]):  # Exclude current position
                    alpha = (i + 1) / len(trace_points)
                    ax.plot(point[0], point[1], 'ro', markersize=4, alpha=alpha * 0.6)
            
            # Mark current expert action with red cross
            # When using origin='upper', we need to flip Y coordinates for plotting
            plot_expert_action = action.copy()
            plot_expert_action[1] = self.action_high[1] - action[1] + self.action_low[1]
            
            ax.plot(plot_expert_action[0], plot_expert_action[1], 'r+', markersize=20, markeredgewidth=4, 
                    label='Expert Action')
        
        # Show policy prediction if requested (in both modes)
        if show_policy_prediction:
            policy_action = self._get_policy_prediction(observation)
            plot_policy_action = policy_action.copy()
            plot_policy_action[1] = self.action_high[1] - policy_action[1] + self.action_low[1]
            
            # Add policy prediction trace (snake-like trail of recent policy predictions)
            if hasattr(self, 'policy_trace') and len(self.policy_trace) > 1:
                policy_trace_points = []
                for i, past_policy_action in enumerate(self.policy_trace[:-1]):  # Exclude current action
                    # Apply coordinate transformation for plotting
                    plot_past_policy_action = past_policy_action.copy()
                    plot_past_policy_action[1] = self.action_high[1] - past_policy_action[1] + self.action_low[1]
                    policy_trace_points.append(plot_past_policy_action)
                    
                    # Draw fading trace points
                    alpha = (i + 1) / len(self.policy_trace)
                    ax.plot(plot_past_policy_action[0], plot_past_policy_action[1], 'o', 
                           color='yellow', markersize=3, alpha=alpha)
                
                # Draw lines connecting policy trace points
                if len(policy_trace_points) > 1:
                    for i in range(len(policy_trace_points) - 1):
                        alpha = (i + 2) / len(self.policy_trace)  # +2 because we excluded current action
                        ax.plot([policy_trace_points[i][0], policy_trace_points[i+1][0]], 
                               [policy_trace_points[i][1], policy_trace_points[i+1][1]], 
                               'y-', alpha=alpha, linewidth=2)
            
            ax.plot(plot_policy_action[0], plot_policy_action[1], 'w*', markersize=15, markeredgewidth=2, 
                    markeredgecolor='black', label='Policy Prediction')
        
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
        ax.set_title(f'{self._get_policy_name()} Action Landscape\n({self.scoring_method.replace("_", " ").title()})', 
                    fontsize=14)
        ax.legend(loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image array
        fig.canvas.draw()
        # Use buffer_rgba() instead of tostring_rgb() for better compatibility
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        # Convert RGBA to RGB
        img_array = img_array[:, :, :3]
        
        plt.close(fig)
        return img_array
    
    def _resize_frame(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        """Resize frame to target height while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))
    
    def _action_to_env_coords(self, action: np.ndarray, env_frame_shape: tuple) -> tuple:
        """Convert action coordinates to environment frame pixel coordinates.
        
        Args:
            action: Action in environment coordinate system [x, y]
            env_frame_shape: Shape of environment frame (height, width, channels)
            
        Returns:
            (pixel_x, pixel_y) coordinates in the environment frame
        """
        # PushT environment bounds (from gym_pusht)
        # The environment renders a 512x512 pixel view of a coordinate system
        # Action space is typically [-1, 1] for both x and y
        env_height, env_width = env_frame_shape[:2]
        
        # Convert action coordinates to pixel coordinates
        # Assuming action space is normalized to [-1, 1] and environment is 512x512
        # Map action [-1, 1] to pixel [0, 512]
        pixel_x = int((action[0] - self.action_low[0]) / (self.action_high[0] - self.action_low[0]) * env_width)
        pixel_y = int((action[1] - self.action_low[1]) / (self.action_high[1] - self.action_low[1]) * env_height)
        
        # Clamp to frame bounds
        pixel_x = max(0, min(env_width - 1, pixel_x))
        pixel_y = max(0, min(env_height - 1, pixel_y))
        
        return pixel_x, pixel_y

    def _overlay_policy_predictions_on_env(self, env_frame: np.ndarray, policy_action: np.ndarray, expert_mode: bool) -> np.ndarray:
        """Overlay policy predictions and traces on environment frame.
        
        Args:
            env_frame: Environment frame to overlay on
            policy_action: Current policy prediction
            expert_mode: Whether we're in expert mode (affects what to show)
            
        Returns:
            Environment frame with overlays
        """
        frame = env_frame.copy()
        
        # Draw policy prediction trace (if exists)
        if hasattr(self, 'policy_trace') and len(self.policy_trace) > 1:
            trace_points = []
            for i, past_action in enumerate(self.policy_trace[:-1]):  # Exclude current action
                pixel_x, pixel_y = self._action_to_env_coords(past_action, frame.shape)
                trace_points.append((pixel_x, pixel_y))
                
                # Draw fading trace points
                alpha = (i + 1) / len(self.policy_trace)
                color_intensity = int(255 * alpha)
                cv2.circle(frame, (pixel_x, pixel_y), 3, (color_intensity, color_intensity, 0), -1)  # Yellow circles
            
            # Draw lines connecting trace points
            for i in range(len(trace_points) - 1):
                alpha = (i + 2) / len(self.policy_trace)  # +2 because we excluded current action
                color_intensity = int(255 * alpha)
                cv2.line(frame, trace_points[i], trace_points[i + 1], (color_intensity, color_intensity, 0), 2)
        
        # Draw current policy prediction as a star
        pixel_x, pixel_y = self._action_to_env_coords(policy_action, frame.shape)
        
        # Draw a star shape (approximated with lines)
        star_size = 8
        # Draw + shape
        cv2.line(frame, (pixel_x - star_size, pixel_y), (pixel_x + star_size, pixel_y), (255, 255, 255), 3)  # White
        cv2.line(frame, (pixel_x, pixel_y - star_size), (pixel_x, pixel_y + star_size), (255, 255, 255), 3)  # White
        # Draw x shape
        cv2.line(frame, (pixel_x - star_size//2, pixel_y - star_size//2), (pixel_x + star_size//2, pixel_y + star_size//2), (255, 255, 255), 2)
        cv2.line(frame, (pixel_x - star_size//2, pixel_y + star_size//2), (pixel_x + star_size//2, pixel_y - star_size//2), (255, 255, 255), 2)
        
        return frame

    def visualize_episode(
        self, 
        episode_idx: Optional[int] = None, 
        output_dir: Path = Path("outputs/policy_landscape_viz"), 
        max_steps: Optional[int] = None,
        show_policy_prediction: bool = True,
        show_confidence_intervals: bool = False,
        target_height: int = 480,
    ) -> None:
        """
        Visualize an episode in two modes:
        1. Expert replay mode (episode_idx specified): replay expert actions.
        2. Policy rollout mode (episode_idx is None): let policy drive actions.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Mode selection ---
        expert_mode = episode_idx is not None

        if expert_mode:
            # --- Expert Replay Mode ---
            # Get episode data from dataset using episode_data_index
            from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
            to_idx = self.dataset.episode_data_index["to"][episode_idx].item()
            episode_data = self.dataset.hf_dataset.select(range(from_idx, to_idx))
            expert_actions = np.array(episode_data["action"])
            if max_steps is not None:
                expert_actions = expert_actions[:max_steps]
                to_idx = min(to_idx, from_idx + max_steps)
            # Load full initial state from raw zarr dataset (includes T-shape position!)
            initial_state = self._load_episode_initial_state(episode_idx)
        else:
            # --- Policy Rollout Mode ---
            expert_actions = None  # Not used
            initial_state = None   # Let env randomize or use default

        # --- Reset environment ---
        obs, _ = self.env.reset()

        if expert_mode:
            # Set the complete initial state from the dataset
            self.env.unwrapped.agent.position = list(initial_state[:2])          # [agent_x, agent_y]
            self.env.unwrapped.block.position = list(initial_state[2:4])         # [block_x, block_y] 
            self.env.unwrapped.block.angle = initial_state[4]                    # block_angle
            self.env.unwrapped.space.step(self.env.unwrapped.dt)
            # Get updated observation after setting complete state
            obs = self.env.unwrapped.get_obs()

        # Reset action trace for new episode
        self._reset_action_trace()
        self._reset_policy_trace()

        # Storage for frames
        left_frames = []  # Environment frames
        right_frames = []  # Action landscape heatmaps
        combined_frames = []

        policy_name = self._get_policy_name()
        mode_str = "Expert Replay" if expert_mode else "Policy Rollout"
        logging.info(f"Generating {policy_name} visualization in {mode_str} mode for episode {episode_idx if expert_mode else 'N/A'}")
        logging.info(f"Scoring method: {self.scoring_method}, Colormap: {self.colormap}")

        # Track current observation for fallback handling
        self.current_observation = None

        # --- Main loop ---
        step_range = range(len(expert_actions)) if expert_mode else range(max_steps or 200)
        for step in tqdm(step_range):
            if expert_mode:
                # Get current expert action
                action = expert_actions[step]
                # Update expert action trace
                self._update_action_trace(action)
            else:
                # Policy rollout: get action from policy
                policy_obs = self._prepare_observation_for_policy(obs)
                action = self._get_policy_prediction(policy_obs)

            # Load the original image from dataset for this step (only in expert mode)
            dataset_image = self._load_episode_image(episode_idx, step) if expert_mode else None

            # Prepare observation for policy (using dataset image if available)
            policy_obs = self._prepare_observation_for_policy(obs, dataset_image)
            self.current_observation = policy_obs  # Store for fallback handling

            # Get policy prediction for visualization (in both modes)
            if show_policy_prediction:
                policy_action = self._get_policy_prediction(policy_obs)
                self._update_policy_trace(policy_action)
            else:
                policy_action = None

            # Render current environment state
            env_frame = self.env.render()
            
            # Overlay policy predictions on environment frame if requested
            if show_policy_prediction and policy_action is not None:
                env_frame = self._overlay_policy_predictions_on_env(env_frame, policy_action, expert_mode)
            
            left_frames.append(env_frame)

            # Create enhanced action landscape heatmap
            heatmap_frame = self._create_enhanced_action_heatmap(
                policy_obs, 
                action,
                expert_mode,
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

            # Step environment
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break

        # Save videos
        logging.info("Saving frame sequences...")
        fps = 10
        env_video_path = output_dir / f"episode_{episode_idx if expert_mode else 'policy'}_environment.mp4"
        imageio.mimsave(str(env_video_path), left_frames, fps=fps)
        heatmap_video_path = output_dir / f"episode_{episode_idx if expert_mode else 'policy'}_heatmaps_{self.scoring_method}.mp4"
        imageio.mimsave(str(heatmap_video_path), right_frames, fps=fps)
        combined_video_path = output_dir / f"episode_{episode_idx if expert_mode else 'policy'}_combined_{self.scoring_method}.mp4"
        imageio.mimsave(str(combined_video_path), combined_frames, fps=fps)
        logging.info(f"Videos saved:")
        logging.info(f"  Environment: {env_video_path}")
        logging.info(f"  Heatmaps: {heatmap_video_path}")
        logging.info(f"  Combined: {combined_video_path}")

        # Save configuration and statistics
        config = {
            "episode_idx": episode_idx if expert_mode else None,
            "num_steps": len(combined_frames),
            "initial_state": {
                "agent_pos": initial_state[:2].tolist() if expert_mode else None,
                "block_pos": initial_state[2:4].tolist() if expert_mode else None,
                "block_angle": float(initial_state[4]) if expert_mode else None
            } if expert_mode else None,
            "action_resolution": self.action_resolution,
            "scoring_method": self.scoring_method,
            "colormap": self.colormap,
            "show_policy_prediction": show_policy_prediction,
            "show_confidence_intervals": show_confidence_intervals,
            "target_height": target_height,
            "policy_type": self._get_policy_name().lower().replace(" ", "_"),
            "mode": mode_str,
            "note": f"{'Complete initial state loaded from zarr dataset, images loaded from HuggingFace video dataset' if expert_mode else 'Policy rollout, initial state randomized or default'}",
        }
        with open(output_dir / f"episode_{episode_idx if expert_mode else 'policy'}_config_{self._get_policy_name().lower().replace(' ', '_')}.json", "w") as f:
            json.dump(config, f, indent=2)

        # Print fallback statistics
        self.print_fallback_statistics()

    def _ensure_tensor_device_dtype(self, tensor: torch.Tensor, target_device: str = None) -> torch.Tensor:
        """Ensure tensor is on correct device with correct dtype for optimal GPU performance."""
        device = target_device or self.device
        
        # Ensure float32 for both MPS and CUDA compatibility/performance
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        
        # Move to device if not already there
        if tensor.device.type != device:
            tensor = tensor.to(device)
            
        return tensor

    def _classify_and_log_error(self, error: Exception, context: str) -> str:
        """Classify error type and update statistics."""
        error_str = str(error).lower()
        error_type = "unknown"
        
        if "out of memory" in error_str or "cuda out of memory" in error_str:
            error_type = "memory_error"
            self.fallback_stats['memory_errors'] += 1
            logging.warning(f"GPU Memory Error in {context}: {error}")
            if self.fallback_stats['memory_errors'] == 1:  # First memory error
                logging.warning("💡 SUGGESTION: Try reducing batch size with --batch_size=50 or lower")
                logging.warning("💡 Current sampling batch size is 50, main batch size is {self.batch_size}")
        elif "device" in error_str or "mps" in error_str or "cuda" in error_str:
            error_type = "device_error"
            self.fallback_stats['device_errors'] += 1
            logging.warning(f"Device Error in {context}: {error}")
        elif "singular" in error_str or "kde" in error_str or "covariance" in error_str:
            error_type = "kde_failure"
            self.fallback_stats['kde_failures'] += 1
            logging.warning(f"KDE/Statistical Error in {context}: {error}")
        elif "policy" in error_str or "select_action" in error_str:
            error_type = "policy_error"
            self.fallback_stats['policy_errors'] += 1
            logging.warning(f"Policy Error in {context}: {error}")
        elif "nan" in error_str or "inf" in error_str or "numerical" in error_str:
            error_type = "numerical_error"
            self.fallback_stats['numerical_errors'] += 1
            logging.warning(f"Numerical Error in {context}: {error}")
        else:
            logging.warning(f"Unclassified Error in {context}: {error}")
            
        return error_type
    
    def _handle_fallback(self, error: Exception, context: str, actions: torch.Tensor) -> torch.Tensor:
        """Handle fallback with comprehensive error tracking and graceful degradation."""
        self.fallback_stats['total_steps'] += 1
        error_type = self._classify_and_log_error(error, context)
        
        # Try "freeze last good state" first
        if self.last_good_scores is not None and self.consecutive_failures < self.max_consecutive_failures:
            self.consecutive_failures += 1
            self.fallback_stats['frozen_steps'] += 1
            logging.info(f"🧊 Using frozen last good state (consecutive failures: {self.consecutive_failures})")
            
            # Ensure the cached scores match current action grid size
            if self.last_good_scores.shape[0] == actions.shape[0]:
                return self.last_good_scores.clone()
            else:
                logging.warning(f"Cached scores shape mismatch: {self.last_good_scores.shape[0]} vs {actions.shape[0]}")
        
        # Fallback to distance-based scoring with normalization
        logging.warning(f"🔄 Falling back to distance-based scoring (failure #{self.consecutive_failures + 1})")
        
        try:
            # Get policy prediction for distance calculation
            if hasattr(self, 'policy') and self.policy is not None:
                single_obs = {k: v[:1] for k, v in self._get_current_observation().items()}
                policy_action = self.policy.select_action(single_obs)
            else:
                # Ultimate fallback: center of action space
                policy_action = torch.tensor([(self.action_low[0] + self.action_high[0]) / 2,
                                            (self.action_low[1] + self.action_high[1]) / 2], 
                                           device=self.device)
            
            # Compute normalized distance scores
            distances = torch.norm(actions - policy_action, dim=1)
            scores = -distances
            
            # Normalize to [0, 1] range for consistency
            if scores.max() != scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = torch.ones_like(scores) * 0.5
                
            return scores
            
        except Exception as fallback_error:
            logging.error(f"❌ Even fallback failed: {fallback_error}")
            # Ultimate fallback: uniform scores
            return torch.ones(actions.shape[0], device=self.device) * 0.5
    
    def _record_successful_computation(self, scores: torch.Tensor):
        """Record successful computation and cache scores for potential future fallback."""
        self.fallback_stats['successful_steps'] += 1
        self.fallback_stats['total_steps'] += 1
        self.consecutive_failures = 0  # Reset failure counter
        self.last_good_scores = scores.clone().detach()  # Cache for future fallback
    
    def _get_current_observation(self) -> Dict[str, torch.Tensor]:
        """Get current observation for fallback handling."""
        if hasattr(self, 'current_observation') and self.current_observation is not None:
            return self.current_observation
        else:
            # Fallback: return empty dict (will trigger ultimate fallback in _handle_fallback)
            return {}
    
    def print_fallback_statistics(self):
        """Print comprehensive fallback statistics."""
        stats = self.fallback_stats
        total = stats['total_steps']
        
        if total == 0:
            logging.info("📊 No fallback events recorded")
            return
            
        logging.info("📊 FALLBACK STATISTICS:")
        logging.info(f"   Total steps processed: {total}")
        logging.info(f"   Successful steps: {stats['successful_steps']} ({stats['successful_steps']/total*100:.1f}%)")
        logging.info(f"   Steps with fallbacks: {total - stats['successful_steps']} ({(total - stats['successful_steps'])/total*100:.1f}%)")
        
        if stats['frozen_steps'] > 0:
            logging.info(f"   🧊 Frozen state fallbacks: {stats['frozen_steps']} ({stats['frozen_steps']/total*100:.1f}%)")
        
        logging.info("   Error breakdown:")
        for error_type, count in stats.items():
            if error_type.endswith('_errors') and count > 0:
                logging.info(f"     {error_type.replace('_', ' ').title()}: {count}")
        
        # Recommendations
        if stats['memory_errors'] > 0:
            logging.info("💡 RECOMMENDATIONS:")
            logging.info("   - Reduce batch size: --batch_size=50 or lower")
            logging.info("   - Reduce sampling batch size in code (currently 50)")
        
        if stats['kde_failures'] > 0:
            logging.info("💡 KDE failures suggest very deterministic policy or numerical issues")
            
        if stats['device_errors'] > 0:
            logging.info("💡 Device errors suggest GPU/MPS compatibility issues")


class DiffusionPolicyLandscapeVisualizer(BasePolicyLandscapeVisualizer):
    """Diffusion Policy specific landscape visualizer."""
    
    def _load_policy(self):
        """Load the Diffusion Policy model."""
        logging.info(f"Loading Diffusion Policy from {self.policy_path}")
        self.policy = DiffusionPolicy.from_pretrained(self.policy_path)
        self.policy.eval()
        self.policy.to(self.device)
    
    def _get_policy_name(self) -> str:
        return "Diffusion Policy"
    
    def _get_policy_prediction(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get the policy's predicted action for the current observation."""
        with torch.no_grad():
            action = self.policy.select_action(observation)
            return action.cpu().numpy().flatten()
    
    def _compute_action_likelihood(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute empirical action probability density using diffusion sampling.
        
        Creates a 2D histogram of action space distribution at current state by:
        1. Sampling N action sequences from the diffusion policy
        2. Taking first action from each sequence  
        3. Building 2D histogram/KDE of (x,y) action components
        4. Returning density values for query actions
        
        This shows what the policy believes about action desirability.
        """
        try:
            # Sample from diffusion policy to build empirical distribution
            n_samples = 500  # Number of diffusion samples (reduced for debugging)
            
            # Create single observation for sampling
            single_obs = {k: v[:1] for k, v in observation.items()}
            
            # Sample action sequences from diffusion policy
            with torch.no_grad():
                sampled_actions = []
                batch_size = 50  # Process in smaller batches to avoid memory issues
                
                for i in range(0, n_samples, batch_size):
                    current_batch_size = min(batch_size, n_samples - i)
                    
                    # Sample one action at a time to avoid queue size mismatches
                    batch_actions = []
                    for j in range(current_batch_size):
                        # Sample single action to avoid queue size mismatches
                        single_action = self.policy.select_action(single_obs)
                        batch_actions.append(single_action)
                    
                    # Stack individual actions
                    actions_batch = torch.stack(batch_actions, dim=0)
                    
                    # Take only first action from each sequence (shape: [batch_size, action_dim])
                    if len(actions_batch.shape) > 2:  # If temporal dimension exists
                        first_actions = actions_batch[:, 0]  # Take first timestep
                    else:
                        first_actions = actions_batch
                    
                    sampled_actions.append(first_actions)
                
                # Concatenate all sampled first actions
                all_sampled_actions = torch.cat(sampled_actions, dim=0)  # [n_samples, 2]
            
            # Build 2D histogram/KDE from sampled actions
            from scipy.stats import gaussian_kde
            
            # Convert to numpy for KDE
            sampled_xy = all_sampled_actions.cpu().numpy()  # [n_samples, 2]
            
            # Debug: Print sampling statistics
            print(f"Sampled {len(sampled_xy)} actions")
            print(f"Action range: X=[{sampled_xy[:, 0].min():.1f}, {sampled_xy[:, 0].max():.1f}], Y=[{sampled_xy[:, 1].min():.1f}, {sampled_xy[:, 1].max():.1f}]")
            print(f"Action std: X={sampled_xy[:, 0].std():.1f}, Y={sampled_xy[:, 1].std():.1f}")
            
            # Create KDE from sampled actions
            if len(sampled_xy) > 10:  # Need minimum samples for KDE
                try:
                    kde = gaussian_kde(sampled_xy.T)  # KDE expects [2, n_samples]
                    
                    # Evaluate KDE at query action points
                    query_xy = actions.cpu().numpy()  # [batch_size, 2]
                    density_scores = kde(query_xy.T)  # Evaluate at query points
                    
                    # Convert back to tensor and normalize
                    scores = torch.from_numpy(density_scores).float().to(self.device)
                    
                    # Debug: Print score statistics
                    print(f"Density scores range: [{scores.min().item():.2e}, {scores.max().item():.2e}]")
                    print(f"Total probability mass over grid: {scores.sum().item():.4f}")
                    
                    # Optional: Normalize over the discrete grid for better visualization
                    # This ensures the discrete grid probabilities sum to 1
                    if scores.sum() > 0:
                        scores = scores / scores.sum()
                        print(f"After grid normalization: [{scores.min().item():.2e}, {scores.max().item():.2e}], sum={scores.sum().item():.4f}")
                    
                    # DON'T log transform - keep raw probabilities for better visualization
                    # scores = torch.log(scores + 1e-8)  # REMOVED
                    
                except Exception as kde_error:
                    logging.warning(f"KDE failed: {kde_error}. Using histogram fallback.")
                    # Fallback to 2D histogram
                    import numpy as np
                    
                    # Create 2D histogram with more bins for better resolution
                    hist, x_edges, y_edges = np.histogram2d(
                        sampled_xy[:, 0], sampled_xy[:, 1], 
                        bins=100, density=True  # Increased bins from 50 to 100
                    )
                    
                    # Debug: Print histogram statistics
                    print(f"Histogram range: [{hist.min():.2e}, {hist.max():.2e}]")
                    print(f"Non-zero bins: {np.count_nonzero(hist)} / {hist.size}")
                    
                    # Interpolate histogram values for query points
                    query_xy = actions.cpu().numpy()
                    
                    # Find bin indices for each query point
                    x_indices = np.digitize(query_xy[:, 0], x_edges) - 1
                    y_indices = np.digitize(query_xy[:, 1], y_edges) - 1
                    
                    # Clip indices to valid range
                    x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
                    y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)
                    
                    # Get histogram values
                    density_scores = hist[x_indices, y_indices]
                    
                    # Convert to tensor
                    scores = torch.from_numpy(density_scores).float().to(self.device)
                    
                    # Optional: Normalize over the discrete grid for better visualization
                    if scores.sum() > 0:
                        scores = scores / scores.sum()
                        print(f"Histogram grid normalization: sum={scores.sum().item():.4f}")
            else:
                # Fallback if insufficient samples
                scores = torch.zeros(actions.shape[0], device=self.device)
            
            return scores
            
        except Exception as e:
            return self._handle_fallback(e, "diffusion_sampling", actions)
        
        # Record successful computation
        self._record_successful_computation(scores)
        return scores


class ACTPolicyLandscapeVisualizer(BasePolicyLandscapeVisualizer):
    """ACT Policy specific landscape visualizer."""
    
    def _load_policy(self):
        """Load the ACT Policy model."""
        logging.info(f"Loading ACT Policy from {self.policy_path}")
        
        # Handle placeholder path for ACT model
        if self.policy_path == "PLACEHOLDER_ACT_MODEL":
            logging.warning("Using placeholder ACT model path.")
            logging.warning("Please replace 'PLACEHOLDER_ACT_MODEL' with your actual ACT model path when available.")
            logging.warning("For example: 'your_username/act_pusht' or local path to trained model")
            # Create a mock policy for now
            self.policy = None
            return
        
        try:
            self.policy = ACTPolicy.from_pretrained(self.policy_path)
            self.policy.eval()
            self.policy.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load ACT policy: {e}")
            logging.warning("Using mock ACT policy for visualization structure")
            self.policy = None
    
    def _get_policy_name(self) -> str:
        return "ACT Policy"
    
    def _get_policy_prediction(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get the policy's predicted action for the current observation."""
        if self.policy is None:
            # Return dummy action for placeholder (center of action space)
            return np.array([256.0, 256.0])
        
        with torch.no_grad():
            action = self.policy.select_action(observation)
            return action.cpu().numpy().flatten()
    
    def _compute_action_likelihood(
        self, 
        observation: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute empirical action probability density using ACT latent sampling.
        
        Creates a 2D histogram of action space distribution at current state by:
        1. Freezing observation o_t
        2. Sampling many latents z ~ N(0, I)  
        3. Decoding each to a_hat_t = π_θ(o_t, z)[0]
        4. Building 2D histogram/KDE of (x,y) action components
        
        This shows what the ACT policy believes about action desirability.
        """
        if self.policy is None:
            # Return dummy scores for placeholder (random but consistent pattern)
            batch_size = actions.shape[0]
            # Create a simple distance-based pattern for demonstration
            center = torch.tensor([256.0, 256.0], device=self.device)
            distances = torch.norm(actions - center, dim=1)
            return -distances * 0.1  # Simple fallback pattern

        try:
            # Sample from ACT policy using latent variable sampling
            n_samples = 500  # Number of latent samples
            
            # Create single observation for sampling
            single_obs = {k: v[:1] for k, v in observation.items()}
            
            # Sample latent variables and decode to actions
            with torch.no_grad():
                sampled_actions = []
                batch_size = 50  # Process in smaller batches
                
                for i in range(0, n_samples, batch_size):
                    current_batch_size = min(batch_size, n_samples - i)
                    
                    # Sample actions using ACT policy - we'll use multiple calls to select_action
                    # This introduces stochasticity through the inherent randomness in the policy
                    batch_actions = []
                    for j in range(current_batch_size):
                        # Each call to select_action may produce different results due to:
                        # 1. Dropout (if enabled during inference)
                        # 2. Sampling in the policy
                        # 3. Internal stochasticity in ACT
                        action = self.policy.select_action(single_obs)
                        batch_actions.append(action)
                    
                    actions_batch = torch.stack(batch_actions, dim=0)
                    
                    # Take only first action from each sequence (shape: [batch_size, action_dim])
                    if len(actions_batch.shape) > 2:  # If temporal dimension exists
                        first_actions = actions_batch[:, 0]  # Take first timestep
                    else:
                        first_actions = actions_batch
                    
                    sampled_actions.append(first_actions)
                
                # Concatenate all sampled first actions
                all_sampled_actions = torch.cat(sampled_actions, dim=0)  # [n_samples, 2]
            
            # Build 2D histogram/KDE from sampled actions (same as diffusion)
            from scipy.stats import gaussian_kde
            
            # Convert to numpy for KDE
            sampled_xy = all_sampled_actions.cpu().numpy()  # [n_samples, 2]
            
            # Debug: Print sampling statistics
            print(f"ACT sampled {len(sampled_xy)} actions")
            print(f"ACT action range: X=[{sampled_xy[:, 0].min():.1f}, {sampled_xy[:, 0].max():.1f}], Y=[{sampled_xy[:, 1].min():.1f}, {sampled_xy[:, 1].max():.1f}]")
            print(f"ACT action std: X={sampled_xy[:, 0].std():.1f}, Y={sampled_xy[:, 1].std():.1f}")
            
            # Create KDE from sampled actions
            if len(sampled_xy) > 10:  # Need minimum samples for KDE
                try:
                    kde = gaussian_kde(sampled_xy.T)  # KDE expects [2, n_samples]
                    
                    # Evaluate KDE at query action points
                    query_xy = actions.cpu().numpy()  # [batch_size, 2]
                    density_scores = kde(query_xy.T)  # Evaluate at query points
                    
                    # Convert back to tensor
                    scores = torch.from_numpy(density_scores).float().to(self.device)
                    
                    # Debug: Print score statistics
                    print(f"ACT density scores range: [{scores.min().item():.2e}, {scores.max().item():.2e}]")
                    print(f"ACT total probability mass over grid: {scores.sum().item():.4f}")
                    
                    # Normalize over the discrete grid for better visualization
                    if scores.sum() > 0:
                        scores = scores / scores.sum()
                        print(f"ACT after grid normalization: [{scores.min().item():.2e}, {scores.max().item():.2e}], sum={scores.sum().item():.4f}")
                    
                except Exception as kde_error:
                    logging.warning(f"ACT KDE failed: {kde_error}. Using histogram fallback.")
                    # Fallback to 2D histogram
                    import numpy as np
                    
                    # Create 2D histogram with more bins for better resolution
                    hist, x_edges, y_edges = np.histogram2d(
                        sampled_xy[:, 0], sampled_xy[:, 1], 
                        bins=100, density=True
                    )
                    
                    # Debug: Print histogram statistics
                    print(f"ACT histogram range: [{hist.min():.2e}, {hist.max():.2e}]")
                    print(f"ACT non-zero bins: {np.count_nonzero(hist)} / {hist.size}")
                    
                    # Interpolate histogram values for query points
                    query_xy = actions.cpu().numpy()
                    
                    # Find bin indices for each query point
                    x_indices = np.digitize(query_xy[:, 0], x_edges) - 1
                    y_indices = np.digitize(query_xy[:, 1], y_edges) - 1
                    
                    # Clip indices to valid range
                    x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
                    y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)
                    
                    # Get histogram values
                    density_scores = hist[x_indices, y_indices]
                    
                    # Convert to tensor
                    scores = torch.from_numpy(density_scores).float().to(self.device)
                    
                    # Normalize over the discrete grid for better visualization
                    if scores.sum() > 0:
                        scores = scores / scores.sum()
                        print(f"ACT histogram grid normalization: sum={scores.sum().item():.4f}")
            else:
                # Fallback if insufficient samples
                scores = torch.zeros(actions.shape[0], device=self.device)
            
            return scores
            
        except Exception as e:
            return self._handle_fallback(e, "act_sampling", actions)
        
        # Record successful computation
        self._record_successful_computation(scores)
        return scores


def create_visualizer(policy_type: str, **kwargs) -> BasePolicyLandscapeVisualizer:
    """Factory function to create the appropriate visualizer based on policy type."""
    visualizers = {
        "diffusion": DiffusionPolicyLandscapeVisualizer,
        "act": ACTPolicyLandscapeVisualizer,
        # Future additions:
        # "tdmpc": TDMPCPolicyLandscapeVisualizer,
        # "vqbet": VQBetPolicyLandscapeVisualizer,
    }
    
    if policy_type not in visualizers:
        raise ValueError(f"Unsupported policy type: {policy_type}. Supported types: {list(visualizers.keys())}")
    
    return visualizers[policy_type](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Multi-Policy Action Landscape Visualization")
    parser.add_argument("--policy_type", type=str, required=True, 
                       choices=["diffusion", "act"], 
                       help="Type of policy to visualize")
    parser.add_argument("--policy_path", type=str, required=True, 
                       help="Path to trained policy")
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/pusht",
                       help="HuggingFace dataset repository ID for training dataset. Must be PushT compatible.")
    parser.add_argument("--episode_idx", type=int, default=None,
                       help="Dataset episode index to visualize (if not provided, runs in policy rollout mode)")
    parser.add_argument("--output_dir", type=str, default="outputs/policy_landscape_viz",
                       help="Output directory for visualization videos")
    parser.add_argument("--action_resolution", type=int, default=50,
                       help="Resolution for action space sampling (e.g., 50 would be for a 50x50 grid)")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of steps to visualize")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run policy on - cuda, mps, or cpu")
    parser.add_argument("--scoring_method", type=str, default="distance",
                       choices=["likelihood", "distance"],
                       help="Method for scoring actions")
    parser.add_argument("--colormap", type=str, default="viridis",
                       help="Matplotlib colormap for heatmap. Default is viridis.")
    parser.add_argument("--show_policy_prediction", type=bool, default=True,
                       help="Whether to show policy's predicted action")
    parser.add_argument("--show_confidence_intervals", type=bool, default=False,
                       help="Whether to show confidence interval contours")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for efficient action evaluation")
    parser.add_argument("--target_height", type=int, default=480,
                       help="Target height for video frames")
    parser.add_argument("--trace_length", type=int, default=10,
                       help="Number of past expert actions to show in trace")
    
    args = parser.parse_args()
    
    init_logging()
    
    # Create appropriate visualizer based on policy type
    visualizer = create_visualizer(
        policy_type=args.policy_type,
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        device=args.device,
        action_resolution=args.action_resolution,
        scoring_method=args.scoring_method,
        colormap=args.colormap,
        batch_size=args.batch_size,
        trace_length=args.trace_length,
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
    
    policy_name = visualizer._get_policy_name()
    logging.info(f"{policy_name} visualization complete!")
    
    # Print final fallback statistics summary
    # Fallback occurs when there is an error during action generation by the policy.
    visualizer.print_fallback_statistics()


if __name__ == "__main__":
    main() 