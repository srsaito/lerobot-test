#!/bin/bash

# Record the start time in seconds since the epoch
start_time=$(date +%s)

echo "Resuming LeRobot training..."
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0fast_base \
--dataset.repo_id=ssaito/koch_test_104 \
--output_dir=outputs/train/pi0fast_koch_test_6 \
--job_name=pi0fast_koch_test_6 \
--policy.device=cuda \
--wandb.enable=false \
--tensorboard.enable=true

echo "Saving model to Hugging Face..."
huggingface-cli upload ssaito/pi0fast_koch_test_6 \
  outputs/train/pi0fast_koch_test_6/checkpoints/last/pretrained_model

# Record the end time 
end_time=$(date +%s) 

# Calculate the duration in seconds
duration=$((end_time - start_time)) 

echo "Resumed training and saving took $duration seconds."

echo "LeRobot script finished!\nTraining and saving took $duration seconds." | mail -s "LeRobot script complete" stevensaito@yahoo.com

