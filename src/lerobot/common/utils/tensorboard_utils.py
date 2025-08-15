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
import logging
import os
from pathlib import Path
from termcolor import colored

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from lerobot.common.constants import PRETRAINED_MODEL_DIR
from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"dataset:{cfg.dataset.repo_id}",
        f"seed:{cfg.seed}",
    ]
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


class TensorBoardLogger:
    """A helper class to log objects using TensorBoard."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.tensorboard
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up TensorBoard
        log_dir = self.cfg.log_dir or str(self.log_dir / "tensorboard")
        run_name = f"{self.job_name}"
        if self.cfg.comment:
            run_name = f"{run_name}_{self.cfg.comment}"
        
        self._writer = SummaryWriter(
            log_dir=os.path.join(log_dir, run_name),
            flush_secs=self.cfg.flush_secs,
        )

        # Note: Skipping hyperparameters logging for now as it can be problematic with complex configs
        print(colored("Logs will be synced with TensorBoard.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(f'tensorboard --logdir={log_dir}', 'yellow', attrs=['bold'])}")

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to TensorBoard."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        model_path = checkpoint_dir / PRETRAINED_MODEL_DIR
        
        # Log model file as artifact
        self._writer.add_text(
            f"checkpoints/{step_id}",
            f"Model checkpoint saved at: {model_path}",
        )

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'TensorBoard logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                )
                continue
            self._writer.add_scalar(f"{mode}/{k}", v, step)

    def log_text(self, tag: str, text: str, step: int):
        self._writer.add_text(tag, text, step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        try:
            # Read video file
            video = torch.load(video_path)
            if isinstance(video, torch.Tensor):
                # Add batch dimension if needed
                if video.dim() == 3:
                    video = video.unsqueeze(0)
                # Add channel dimension if needed
                if video.dim() == 4:
                    video = video.unsqueeze(1)
                self._writer.add_video(f"{mode}/video", video, step, fps=self.env_fps)
            else:
                logging.warning(f"Video at {video_path} is not a tensor")
        except Exception as e:
            logging.warning(f"Failed to log video: {e}")

    def close(self):
        """Close the TensorBoard writer."""
        self._writer.close() 
