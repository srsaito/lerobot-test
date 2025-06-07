#!/bin/bash

# Record the start time in seconds since the epoch
start_time=$(date +%s)

echo "Resuming LeRobot training..."
python lerobot/scripts/train.py \
--policy.type=vqbet \
--dataset.repo_id=ssaito/koch_test_104 \
--output_dir=outputs/train/vqbet_koch_test \
--job_name=vqbet_koch_test \
--policy.device=cuda \
--wandb.enable=false \
--tensorboard.enable=true


echo "Saving model to Hugging Face..."
huggingface-cli upload ssaito/vqbet_koch_test \
  outputs/train/vqbet_koch_test/checkpoints/last/pretrained_model

# Record the end time 
end_time=$(date +%s) 

# Calculate the duration in seconds
duration=$((end_time - start_time)) 

echo "Resumed training and saving took $duration seconds."

echo "LeRobot script finished!\nTraining and saving took $duration seconds." | mail -s "LeRobot script complete" stevensaito@yahoo.com

