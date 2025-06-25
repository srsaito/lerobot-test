#!/usr/bin/env python3
"""
Action Space Visualizer for LeRobot Policies

Creates side-by-side visualization of:
- Left: Agent interacting in PushT environment (expert replay or policy rollout)
- Right: 2D action space heat map showing policy's probability distribution

Supports: Diffusion Policy, ACT, VQ-BET
Based on the proven structure of examples/2_evaluate_pretrained_policy.py

Usage:
    # Expert replay mode (episode_num triggers this mode)
    python viz_action_space.py --policy_type diffusion --policy_path lerobot/diffusion_pusht --episode_num 5
    
    # Policy rollout mode (no episode_num)
    python viz_action_space.py --policy_type diffusion --policy_path lerobot/diffusion_pusht --time_steps 100
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Circle
from scipy.stats import entropy, gaussian_kde
from tqdm import tqdm

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

SUPPORTED_POLICIES = ["diffusion", "act", "vqbet"]
POLICY_CLASSES = {
    "diffusion": DiffusionPolicy,
    "act": ACTPolicy, 
    "vqbet": VQBeTPolicy,
}
POLICY_PATHS = {
    "diffusion": "lerobot/diffusion_pusht",
    "act": "ssaito/act_pusht_test", 
    "vqbet": "lerobot/vqbet_pusht",
}

COLUMBIA_DATASET_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
CACHE_DIR = Path.home() / ".cache" / "lerobot" / "pusht_original"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    logger = logging.getLogger("viz_action_space")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / "logs.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def download_columbia_dataset(cache_dir: Path, logger: logging.Logger) -> bool:
    """Download and extract Columbia dataset if not already cached."""
    # Check if already downloaded by looking for the episode_ends directory
    episode_ends_dir = cache_dir / "pusht" / "pusht_cchi_v7_replay.zarr" / "meta" / "episode_ends"
    if episode_ends_dir.exists():
        logger.info(f"Columbia dataset already cached at {cache_dir}")
        return True
    
    logger.info(f"Downloading Columbia dataset to {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download zip file
        zip_path = cache_dir / "pusht.zip"
        urllib.request.urlretrieve(COLUMBIA_DATASET_URL, zip_path)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # Remove zip file
        zip_path.unlink()
        
        # Verify extraction
        episode_ends_dir = cache_dir / "pusht" / "pusht_cchi_v7_replay.zarr" / "meta" / "episode_ends"
        if not episode_ends_dir.exists():
            raise FileNotFoundError("Episode ends directory not found after extraction")
            
        logger.info("Columbia dataset downloaded and extracted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Columbia dataset: {e}")
        return False


def load_columbia_episode_initial_state(episode_idx: int, cache_dir: Path, logger: logging.Logger) -> np.ndarray:
    """Load initial T-block state from Columbia dataset."""
    try:
        # Load episode data - zarr format stores data sequentially with episode boundaries
        import zarr
        
        data_path = cache_dir / "pusht"
        if not data_path.exists():
            raise FileNotFoundError(f"Columbia dataset not found at {data_path}")
        
        # Load the zarr dataset
        zarr_path = data_path / "pusht_cchi_v7_replay.zarr"
        if not zarr_path.exists():
            # Try alternative path structures
            zarr_files = list(data_path.glob("*.zarr"))
            if zarr_files:
                zarr_path = zarr_files[0]
            else:
                raise FileNotFoundError("No zarr files found in Columbia dataset")
        
        root = zarr.open(str(zarr_path), mode='r')
        
        # Get episode boundaries
        episode_ends = root['meta']['episode_ends'][:]
        
        # Calculate the starting index for this episode
        if episode_idx == 0:
            episode_start_idx = 0
        else:
            episode_start_idx = episode_ends[episode_idx - 1]
        
        # Validate episode index
        if episode_idx >= len(episode_ends):
            raise ValueError(f"Episode {episode_idx} not found. Dataset has {len(episode_ends)} episodes (0-{len(episode_ends)-1})")
        
        # Load initial state for the episode: [agent_x, agent_y, block_x, block_y, block_angle]
        states = root['data']['state']
        initial_state = states[episode_start_idx]
        
        logger.info(f"Loaded Columbia episode {episode_idx} initial state: {initial_state}")
        return initial_state
        
    except Exception as e:
        logger.error(f"Failed to load Columbia episode {episode_idx}: {e}")
        raise


def calculate_entropy(probability_grid: np.ndarray) -> float:
    """Calculate entropy of probability distribution."""
    # Flatten and normalize
    probs = probability_grid.flatten()
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    return entropy(probs, base=2)


def format_time_remaining(elapsed_time: float, current_step: int, total_steps: int) -> str:
    """Calculate and format estimated time remaining."""
    if current_step == 0:
        return "Estimating..."
    
    avg_time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_seconds = avg_time_per_step * remaining_steps
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}m"
    else:
        return f"{remaining_seconds/3600:.1f}h"


# ============================================================================
# POLICY VISUALIZER CLASSES
# ============================================================================

class PolicyActionSpaceVisualizer:
    """Main visualizer class handling all policy types."""
    
    def __init__(
        self,
        policy_type: str,
        policy_path: str,
        episode_num: Optional[int] = None,
        time_steps: int = 150,
        action_space_res: int = 12,
        n_samples: int = 24,
        output_path: Optional[str] = None,
        past_trace_len: int = 5,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.episode_num = episode_num
        self.time_steps = time_steps
        self.action_space_res = action_space_res
        self.n_samples = n_samples
        self.past_trace_len = past_trace_len
        self.device = device or get_device()
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine mode
        self.expert_mode = episode_num is not None
        
        # Setup output directory
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"outputs/{policy_type}_viz_{timestamp}"
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.policy = None
        self.dataset = None
        self.action_bounds = None
        self.action_grid = None
        self.past_actions = []
        
        # Timing and statistics
        self.start_time = None
        self.step_times = []
        
        self.logger.info(f"Initialized {policy_type} visualizer")
        self.logger.info(f"Mode: {'Expert Replay' if self.expert_mode else 'Policy Rollout'}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output: {self.output_dir}")
        
    def setup(self):
        """Initialize environment, policy, and datasets."""
        self.logger.info("Setting up environment, policy, and datasets...")
        
        # Setup environment
        self._setup_environment()
        
        # Setup policy
        self._setup_policy()
        
        # Setup datasets
        self._setup_datasets()
        
        # Create action grid
        self._create_action_grid()
        
        self.logger.info("Setup completed successfully")
    
    def _setup_environment(self):
        """Initialize PushT environment."""
        try:
            import gym_pusht  # noqa: F401
            
            self.env = gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                max_episode_steps=300,
            )
            
            # Get action space bounds
            self.action_bounds = {
                'low': self.env.action_space.low,
                'high': self.env.action_space.high
            }
            
            self.logger.info(f"Environment initialized. Action bounds: {self.action_bounds}")
            
        except ImportError as e:
            self.logger.error("gym_pusht not installed. Install with: pip install -e '.[pusht]'")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to setup environment: {e}")
            raise
    
    def _setup_policy(self):
        """Load and initialize policy."""
        try:
            # Handle placeholder models
            if "PLACEHOLDER" in self.policy_path:
                self.logger.warning(f"Using placeholder for {self.policy_type} policy")
                self.policy = None
                return
            
            # Check if model path exists (for local paths)
            if not self.policy_path.startswith("lerobot/") and not self.policy_path.startswith("ssaito/"):
                if not Path(self.policy_path).exists():
                    raise FileNotFoundError(f"Policy path not found: {self.policy_path}")
            
            # Load policy
            policy_class = POLICY_CLASSES[self.policy_type]
            self.policy = policy_class.from_pretrained(self.policy_path)
            self.policy.to(self.device)
            self.policy.eval()
            
            self.logger.info(f"Policy loaded: {self.policy_type} from {self.policy_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
    
    def _setup_datasets(self):
        """Load LeRobot dataset and Columbia dataset if needed."""
        try:
            # Load LeRobot dataset
            self.dataset = LeRobotDataset("lerobot/pusht", download_videos=True)
            self.logger.info(f"LeRobot dataset loaded. Episodes: {len(self.dataset.episode_data_index)}")
            
            # Validate episode number if in expert mode
            if self.expert_mode:
                max_episodes = len(self.dataset.episode_data_index['from'])
                if self.episode_num >= max_episodes:
                    raise ValueError(f"Episode {self.episode_num} not found. Dataset has {max_episodes} episodes (0-{max_episodes-1})")
                
                # Download Columbia dataset for initial T-block states
                if not download_columbia_dataset(CACHE_DIR, self.logger):
                    raise RuntimeError("Failed to download Columbia dataset")
            
        except Exception as e:
            self.logger.error(f"Failed to setup datasets: {e}")
            raise
    
    def _create_action_grid(self):
        """Create grid of actions for heat map sampling."""
        x_actions = np.linspace(
            self.action_bounds['low'][0], 
            self.action_bounds['high'][0], 
            self.action_space_res
        )
        y_actions = np.linspace(
            self.action_bounds['low'][1], 
            self.action_bounds['high'][1], 
            self.action_space_res
        )
        
        xx, yy = np.meshgrid(x_actions, y_actions)
        actions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        self.action_grid = torch.from_numpy(actions).float().to(self.device)
        self.logger.info(f"Action grid created: {self.action_grid.shape}")
    
    def _prepare_observation_for_policy(self, obs_dict: Dict) -> Dict[str, torch.Tensor]:
        """Convert environment observation to policy input format."""
        # Based on examples/2_evaluate_pretrained_policy.py
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
    
    def _sample_policy_actions(self, observation: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Sample multiple actions from policy for heat map generation."""
        if self.policy is None:
            # Return dummy actions for placeholder policies
            dummy_actions = []
            for _ in range(self.n_samples):
                action = np.random.uniform(
                    self.action_bounds['low'], 
                    self.action_bounds['high']
                )
                dummy_actions.append(action)
            return dummy_actions
        
        sampled_actions = []
        
        # Sample actions using the unified select_action interface
        with torch.no_grad():
            for _ in range(self.n_samples):
                action = self.policy.select_action(observation)
                action_np = action.cpu().numpy().flatten()
                sampled_actions.append(action_np)
        
        return sampled_actions
    
    def _create_heat_map(self, observation: Dict[str, torch.Tensor], current_action: np.ndarray, step: int) -> np.ndarray:
        """Create action space heat map showing policy's probability distribution."""
        # Sample actions from policy
        sampled_actions = self._sample_policy_actions(observation)
        
        # Create 2D histogram
        action_array = np.array(sampled_actions)
        
        # Create bins for histogram
        x_edges = np.linspace(self.action_bounds['low'][0], self.action_bounds['high'][0], self.action_space_res + 1)
        y_edges = np.linspace(self.action_bounds['low'][1], self.action_bounds['high'][1], self.action_space_res + 1)
        
        # Create histogram
        hist, _, _ = np.histogram2d(
            action_array[:, 0], action_array[:, 1],
            bins=[x_edges, y_edges]
        )
        
        # Normalize to probability distribution
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        
        # Calculate entropy for display
        heat_map_entropy = calculate_entropy(hist)
        
        # Create visualization with minimal margins to maximize heat map size
        fig, ax = plt.subplots(figsize=(10, 8))  # Wider figure to better match env frame
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # Maximize heat map area
        
        # Create heat map with proper coordinate system (matching PushT's inverted Y)
        im = ax.imshow(
            hist.T,  # TRANSPOSE the histogram to match imshow's coordinate system
            extent=[self.action_bounds['low'][0], self.action_bounds['high'][0], 
                   self.action_bounds['low'][1], self.action_bounds['high'][1]],
            origin='upper',  # Use upper origin to match PushT's coordinate system
            cmap='viridis',
            aspect='auto',  # Allow automatic aspect ratio adjustment to fill space
            alpha=0.8
        )
        
        # Add colorbar with balanced sizing (matching multi-policy script approach)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Probability Density', rotation=270, labelpad=20)
        
        # Mark current action with red cross (flip Y coordinate to match PushT's inverted system)
        plot_action = current_action.copy()
        plot_action[1] = self.action_bounds['high'][1] - current_action[1] + self.action_bounds['low'][1]
        ax.plot(plot_action[0], plot_action[1], 'r+', markersize=24, markeredgewidth=5)
        
        # Draw past action trace (flip Y coordinates to match PushT's inverted system)
        if len(self.past_actions) > 1:
            past_array = np.array(self.past_actions)
            # Apply coordinate transformation to all past actions
            plot_past_array = past_array.copy()
            plot_past_array[:, 1] = self.action_bounds['high'][1] - past_array[:, 1] + self.action_bounds['low'][1]
            
            for i in range(len(plot_past_array) - 1):
                alpha = (i + 1) / len(plot_past_array)
                ax.plot(plot_past_array[i:i+2, 0], plot_past_array[i:i+2, 1], 'r-', alpha=alpha, linewidth=2)
            
            # Mark past positions
            for i, past_action in enumerate(plot_past_array[:-1]):
                alpha = (i + 1) / len(plot_past_array)
                ax.plot(past_action[0], past_action[1], 'ro', markersize=4, alpha=alpha)
        
        # Add step and entropy info with larger font for better readability
        info_text = f"Step: {step}\nEntropy: {heat_map_entropy:.2f}"
        ax.text(0.01, 0.99, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=12, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Remove axis labels and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(self.action_bounds['low'][0], self.action_bounds['high'][0])
        ax.set_ylim(self.action_bounds['low'][1], self.action_bounds['high'][1])
        
        # Convert to image array
        fig.canvas.draw()
        # Use buffer_rgba() and convert to RGB for macOS compatibility
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img_array = img_array[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return img_array
    
    def _resize_frame(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        """Resize frame to target height while maintaining aspect ratio."""
        import cv2
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))
    
    def _combine_frames(self, env_frame: np.ndarray, heat_map_frame: np.ndarray, step: int) -> np.ndarray:
        """Combine environment and heat map frames side-by-side with balanced sizing."""
        import cv2
        
        # Use the same approach as visualize_policy_landscape_multi.py
        # Resize both frames to the same target height while maintaining aspect ratio
        target_height = 480
        
        # Resize environment frame to target height
        env_frame_resized = self._resize_frame(env_frame, target_height)
        
        # Resize heat map frame to target height  
        heat_map_frame_resized = self._resize_frame(heat_map_frame, target_height)
        
        # Combine horizontally
        combined = np.hstack([env_frame_resized, heat_map_frame_resized])
        
        # Add title
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined)
        ax.axis('off')
        
        # Add title spanning both frames
        mode_text = f"Expert Replay (Episode {self.episode_num})" if self.expert_mode else "Policy Rollout"
        title = f"{self.policy_type.title()} Policy - {mode_text}"
        ax.set_title(title, fontsize=22, pad=20)
        
        # Convert to image array
        fig.canvas.draw()
        # Use buffer_rgba() and convert to RGB for macOS compatibility
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img_array = img_array[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return img_array
    
    def run_visualization(self):
        """Main visualization loop."""
        self.logger.info("Starting visualization...")
        self.start_time = time.time()
        
        # Initialize video writer for incremental creation
        video_path = self.output_dir / "video.mp4"
        video_writer = None
        
        try:
            if self.expert_mode:
                frames = self._run_expert_replay()
            else:
                frames = self._run_policy_rollout()
            
            # Create video incrementally
            if frames:
                fps = 4  # 4 fps for smoother playback
                video_writer = imageio.get_writer(str(video_path), fps=fps)
                
                for frame in frames:
                    video_writer.append_data(frame)
                
                video_writer.close()
                self.logger.info(f"Video saved to {video_path}")
            
            # Save configuration and results
            self._save_results()
            
        except KeyboardInterrupt:
            self.logger.info("Visualization interrupted by user")
            if video_writer:
                video_writer.close()
                self.logger.info(f"Partial video saved to {video_path}")
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            if video_writer:
                video_writer.close()
            raise
    
    def _run_expert_replay(self) -> List[np.ndarray]:
        """Run expert replay mode."""
        self.logger.info(f"Running expert replay for episode {self.episode_num}")
        
        # Load episode data from LeRobot dataset
        from_idx = self.dataset.episode_data_index["from"][self.episode_num].item()
        to_idx = self.dataset.episode_data_index["to"][self.episode_num].item()
        episode_length = to_idx - from_idx
        
        # For expert replay, use full episode length (ignore --time_steps limit)
        # Only policy rollout mode should respect --time_steps
        self.logger.info(f"Expert episode {self.episode_num} has {episode_length} steps (using full length)")
        
        episode_data = self.dataset.hf_dataset.select(range(from_idx, to_idx))
        expert_actions = np.array(episode_data["action"])
        
        # Load initial state from Columbia dataset
        try:
            initial_state = load_columbia_episode_initial_state(self.episode_num, CACHE_DIR, self.logger)
        except Exception as e:
            self.logger.warning(f"Failed to load Columbia initial state: {e}. Using environment default.")
            initial_state = None
        
        # Reset environment and policy
        obs, _ = self.env.reset()
        if self.policy:
            self.policy.reset()
        
        # Set initial state if available
        if initial_state is not None:
            # Set agent position from LeRobot dataset
            first_frame = self.dataset[from_idx]
            agent_pos = first_frame["observation.state"].numpy()
            self.env.unwrapped.agent.position = agent_pos.tolist()
            
            # Set T-block state from Columbia dataset (non-legacy mode: angle first, then position)
            self.env.unwrapped.block.angle = initial_state[4]
            self.env.unwrapped.block.position = initial_state[2:4].tolist()
            self.env.unwrapped.space.step(self.env.unwrapped.dt)
            obs = self.env.unwrapped.get_obs()
        
        # Run episode
        frames = []
        total_reward = 0
        
        # Create initial frame (step 0) before any actions are taken
        env_frame_initial = self.env.render()
        
        # For step 0, we'll show the first action that will be taken, but with empty past actions
        if episode_length > 0:
            initial_action = expert_actions[0]
            
            # Use environment's original cross rendering
            # (removed overlay call)
            
            observation_initial = self._prepare_observation_for_policy(obs)
            
            # Create heat map for the first action (with empty past actions trace)
            progress_desc = f"Step 0/{episode_length}: Sampling actions"
            with tqdm(total=self.n_samples, desc=progress_desc, leave=False) as pbar:
                original_sample = self._sample_policy_actions
                
                def sample_with_progress(obs):
                    actions = []
                    for i in range(self.n_samples):
                        if self.policy is None:
                            action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
                        else:
                            with torch.no_grad():
                                action = self.policy.select_action(obs)
                                action = action.cpu().numpy().flatten()
                        actions.append(action)
                        pbar.update(1)
                    return actions
                
                self._sample_policy_actions = sample_with_progress
                heat_map_frame_initial = self._create_heat_map(observation_initial, initial_action, 0)
                self._sample_policy_actions = original_sample
            
            # Combine initial frame
            combined_frame_initial = self._combine_frames(env_frame_initial, heat_map_frame_initial, 0)
            frames.append(combined_frame_initial)
        
        for step in range(episode_length):
            step_start_time = time.time()
            
            # Get expert action
            expert_action = expert_actions[step]
            
            # Step environment FIRST
            obs, reward, terminated, truncated, info = self.env.step(expert_action)
            total_reward += reward
            
            # Update past actions trace AFTER stepping (so the trace matches what we see)
            self.past_actions.append(expert_action.copy())
            if len(self.past_actions) > self.past_trace_len:
                self.past_actions.pop(0)
            
            # Get environment frame (now shows agent at new position after the action)
            env_frame = self.env.render()
            
            # Use environment's original cross rendering
            # (removed overlay call)
            
            # Create heat map with progress bar (showing the action that was just taken)
            observation = self._prepare_observation_for_policy(obs)
            progress_desc = f"Step {step+1}/{episode_length}: Sampling actions"
            
            with tqdm(total=self.n_samples, desc=progress_desc, leave=False) as pbar:
                # Override the sampling method to show progress
                original_sample = self._sample_policy_actions
                
                def sample_with_progress(obs):
                    actions = []
                    for i in range(self.n_samples):
                        if self.policy is None:
                            action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
                        else:
                            with torch.no_grad():
                                action = self.policy.select_action(obs)
                                action = action.cpu().numpy().flatten()
                        actions.append(action)
                        pbar.update(1)
                    return actions
                
                self._sample_policy_actions = sample_with_progress
                heat_map_frame = self._create_heat_map(observation, expert_action, step + 1)
                self._sample_policy_actions = original_sample
            
            # Combine frames
            combined_frame = self._combine_frames(env_frame, heat_map_frame, step + 1)
            frames.append(combined_frame)
            
            # Calculate timing
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            elapsed_time = time.time() - self.start_time
            time_remaining = format_time_remaining(elapsed_time, step + 1, episode_length)
            
            # Log progress
            self.logger.info(f"Step {step+1}/{episode_length} - Reward: {reward:.3f} - "
                           f"Terminated: {terminated} - Time remaining: {time_remaining}")
            
            if terminated or truncated:
                self.logger.info(f"Episode ended at step {step+1}")
                break
        
        self.logger.info(f"Expert replay completed. Total reward: {total_reward:.3f}")
        return frames
    
    def _run_policy_rollout(self) -> List[np.ndarray]:
        """Run policy rollout mode."""
        self.logger.info(f"Running policy rollout for {self.time_steps} steps")
        
        if self.policy is None:
            self.logger.warning("Using placeholder policy - actions will be random")
        
        # Reset environment and policy
        obs, _ = self.env.reset()
        if self.policy:
            self.policy.reset()
        
        frames = []
        total_reward = 0
        
        # Create initial frame (step 0) before any actions are taken
        env_frame_initial = self.env.render()
        
        # For step 0, show what the policy would do from initial state
        observation_initial = self._prepare_observation_for_policy(obs)
        if self.policy is None:
            initial_action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
        else:
            with torch.no_grad():
                action_tensor = self.policy.select_action(observation_initial)
                initial_action = action_tensor.cpu().numpy().flatten()
        
        # Use environment's original cross rendering
        # (removed overlay call)
        
        # Create heat map for initial state
        progress_desc = f"Step 0/{self.time_steps}: Sampling actions"
        with tqdm(total=self.n_samples, desc=progress_desc, leave=False) as pbar:
            original_sample = self._sample_policy_actions
            
            def sample_with_progress(obs):
                actions = []
                for i in range(self.n_samples):
                    if self.policy is None:
                        action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
                    else:
                        with torch.no_grad():
                            action = self.policy.select_action(obs)
                            action = action.cpu().numpy().flatten()
                    actions.append(action)
                    pbar.update(1)
                return actions
            
            self._sample_policy_actions = sample_with_progress
            heat_map_frame_initial = self._create_heat_map(observation_initial, initial_action, 0)
            self._sample_policy_actions = original_sample
        
        # Combine initial frame
        combined_frame_initial = self._combine_frames(env_frame_initial, heat_map_frame_initial, 0)
        frames.append(combined_frame_initial)
        
        for step in range(self.time_steps):
            step_start_time = time.time()
            
            # Get policy action
            observation = self._prepare_observation_for_policy(obs)
            
            if self.policy is None:
                # Placeholder action
                policy_action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
            else:
                with torch.no_grad():
                    action_tensor = self.policy.select_action(observation)
                    policy_action = action_tensor.cpu().numpy().flatten()
            
            # Step environment FIRST
            obs, reward, terminated, truncated, info = self.env.step(policy_action)
            total_reward += reward
            
            # Update past actions trace AFTER stepping
            self.past_actions.append(policy_action.copy())
            if len(self.past_actions) > self.past_trace_len:
                self.past_actions.pop(0)
            
            # Get environment frame (now shows agent at new position)
            env_frame = self.env.render()
            
            # Use environment's original cross rendering
            # (removed overlay call)
            
            # Create heat map with progress bar (showing the action that was just taken)
            progress_desc = f"Step {step+1}/{self.time_steps}: Sampling actions"
            
            with tqdm(total=self.n_samples, desc=progress_desc, leave=False) as pbar:
                # Override the sampling method to show progress
                original_sample = self._sample_policy_actions
                
                def sample_with_progress(obs):
                    actions = []
                    for i in range(self.n_samples):
                        if self.policy is None:
                            action = np.random.uniform(self.action_bounds['low'], self.action_bounds['high'])
                        else:
                            with torch.no_grad():
                                action = self.policy.select_action(obs)
                                action = action.cpu().numpy().flatten()
                        actions.append(action)
                        pbar.update(1)
                    return actions
                
                self._sample_policy_actions = sample_with_progress
                heat_map_frame = self._create_heat_map(observation, policy_action, step + 1)
                self._sample_policy_actions = original_sample
            
            # Combine frames
            combined_frame = self._combine_frames(env_frame, heat_map_frame, step + 1)
            frames.append(combined_frame)
            
            # Calculate timing
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            elapsed_time = time.time() - self.start_time
            time_remaining = format_time_remaining(elapsed_time, step + 1, self.time_steps)
            
            # Log progress
            self.logger.info(f"Step {step+1}/{self.time_steps} - Reward: {reward:.3f} - "
                           f"Terminated: {terminated} - Time remaining: {time_remaining}")
            
            if terminated:
                self.logger.info(f"Episode succeeded at step {step+1}!")
                break
            elif truncated:
                self.logger.info(f"Episode truncated at step {step+1}")
                break
        
        self.logger.info(f"Policy rollout completed. Total reward: {total_reward:.3f}")
        return frames
    
    def _save_results(self):
        """Save configuration and results to files."""
        # Save configuration
        config = {
            "policy_type": self.policy_type,
            "policy_path": self.policy_path,
            "episode_num": self.episode_num,
            "time_steps": self.time_steps,
            "action_space_res": self.action_space_res,
            "n_samples": self.n_samples,
            "past_trace_len": self.past_trace_len,
            "device": self.device,
            "expert_mode": self.expert_mode,
            "output_dir": str(self.output_dir),
        }
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save results
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        results = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.now().isoformat(),
            "total_execution_time_seconds": total_time,
            "average_step_time_seconds": avg_step_time,
            "steps_completed": len(self.step_times),
            "action_bounds": {
                "low": self.action_bounds['low'].tolist(),
                "high": self.action_bounds['high'].tolist()
            } if self.action_bounds else None,
        }
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
        self.logger.info(f"Results saved to {results_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def validate_args(args):
    """Validate command line arguments."""
    errors = []
    
    # Check policy type
    if args.policy_type not in SUPPORTED_POLICIES:
        errors.append(f"Unsupported policy type: {args.policy_type}. Supported: {SUPPORTED_POLICIES}")
    
    # Check for incompatible combinations
    if args.episode_num is not None and args.episode_num < 0:
        errors.append("episode_num must be non-negative")
    
    if args.time_steps <= 0:
        errors.append("time_steps must be positive")
    
    if args.action_space_res <= 0:
        errors.append("action_space_res must be positive")
    
    if args.n_samples <= 0:
        errors.append("n_samples must be positive")
    
    if args.past_trace_len < 0:
        errors.append("past_trace_len must be non-negative")
    
    # Resource considerations (informational)
    total_samples = args.time_steps * args.n_samples
    if total_samples > 50000:
        print(f"Warning: Large number of total samples ({total_samples}). This may take a long time.")
    
    if errors:
        for error in errors:
            print(f"Error: {error}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Action Space Visualizer for LeRobot Policies")
    
    # Required arguments
    parser.add_argument("--policy_type", type=str, required=True, 
                       choices=SUPPORTED_POLICIES,
                       help="Type of policy to visualize")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to trained policy (HF repo or local path)")
    
    # Mode selection
    parser.add_argument("--episode_num", type=int, default=None,
                       help="Episode number for expert replay mode (0-205). If not provided, runs policy rollout mode.")
    
    # Visualization parameters
    parser.add_argument("--time_steps", type=int, default=150,
                       help="Number of simulation time steps to run")
    parser.add_argument("--action_space_res", type=int, default=12,
                       help="Resolution of action/heat map (NxN grid)")
    parser.add_argument("--n_samples", type=int, default=24,
                       help="Number of samples for heat map generation")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output directory path (default: auto-generated)")
    parser.add_argument("--past_trace_len", type=int, default=5,
                       help="Number of past actions to show in trace")
    
    # System parameters
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/mps/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Create output directory for logging
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"outputs/{args.policy_type}_viz_{timestamp}"
    else:
        output_path = args.output_path
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        # Create and run visualizer
        visualizer = PolicyActionSpaceVisualizer(
            policy_type=args.policy_type,
            policy_path=args.policy_path,
            episode_num=args.episode_num,
            time_steps=args.time_steps,
            action_space_res=args.action_space_res,
            n_samples=args.n_samples,
            output_path=output_path,
            past_trace_len=args.past_trace_len,
            device=args.device,
            logger=logger,
        )
        
        # Setup and run
        visualizer.setup()
        visualizer.run_visualization()
        
        logger.info("Visualization completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 