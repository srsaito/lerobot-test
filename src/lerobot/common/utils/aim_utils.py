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

from aim import Run, Image, Video
from aim.sdk.objects.distribution import Distribution

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


def get_safe_aim_artifact_name(name: str):
    """AIM artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class AimLogger:
    """A helper class to log objects using AIM."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.aim
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up AIM
        self._aim = Run(
            experiment=self.cfg.experiment_name,
            repo=self.cfg.repo,
            run_hash=self.cfg.run_hash,
            description=self.cfg.notes,
            log_system_params=True,
        )

        # Log configuration
        self._aim["config"] = cfg.to_dict()
        self._aim["group"] = self._group

        print(colored("Logs will be synced with AIM.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(self._aim.get_url(), 'yellow', attrs=['bold'])}")

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to AIM."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_aim_artifact_name(artifact_name)
        
        # Log model file as artifact
        model_path = checkpoint_dir / PRETRAINED_MODEL_DIR
        self._aim.track_file(model_path, artifact_name)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'AIM logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                )
                continue
            self._aim.track(v, name=f"{mode}/{k}", step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        aim_video = Video(video_path, fps=self.env_fps)
        self._aim.track(aim_video, name=f"{mode}/video", step=step)

    def close(self):
        """Close the AIM run."""
        self._aim.close() 