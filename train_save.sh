#!/bin/bash

# Record the start time in seconds since the epoch
start_time=$(date +%s)

echo "Resuming LeRobot training..."
python lerobot/scripts/train.py \
    --config_path=outputs/train/pi0_koch_test_6/checkpoints/last/pretrained_model/ \
    --resume=true

echo "Saving model to Hugging Face..."
huggingface-cli upload ssaito/pi0_koch_test_6 \
  outputs/train/pi0_koch_test_6/checkpoints/last/pretrained_model

# Record the end time 
end_time=$(date +%s) 

# Calculate the duration in seconds
duration=$((end_time - start_time)) 

echo "Resumed training and saving took $duration seconds."

echo "LeRobot script finished!\nTraining and saving took $duration seconds." | mail -s "LeRobot script complete" stevensaito@yahoo.com

