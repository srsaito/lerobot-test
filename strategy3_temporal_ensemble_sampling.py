"""
Strategy 3: Proper Temporal Ensembling for Heatmap Sampling

This approach maintains the FIFO buffer state across samples while re-sampling z
for each new sample, as required for heatmap generation.
"""

import torch
import copy
from typing import List
import numpy as np

def sample_act_actions_with_proper_temporal_ensembling(
    policy, 
    observation, 
    n_samples: int,
    current_timestep: int = 3  # Assuming we're at t=3 like in Figure 5
) -> List[np.ndarray]:
    """
    Sample actions for heatmap while preserving temporal ensembling FIFO buffer.
    
    Key insight: We need to preserve the historical chunks (t=0,1,2) while only
    re-sampling the current timestep chunk (t=3) with different z values.
    """
    
    # Ensure temporal ensembling is enabled
    if policy.config.temporal_ensemble_coeff is None:
        raise ValueError("Temporal ensembling must be enabled for this sampling strategy")
    
    # Step 1: Build up the historical FIFO buffer state by running forward to current_timestep
    policy.reset()  # Start fresh
    
    # Simulate the historical timesteps to build up the FIFO buffer
    # This establishes the "context" that should remain constant across samples
    for t in range(current_timestep):
        # Use the same observation for historical context
        # In practice, you might want to use actual historical observations
        _ = policy.select_action(observation)
    
    # Step 2: Save the current ensembler state (this is our "historical context")
    # We need to preserve: ensembled_actions and ensembled_actions_count
    historical_ensembled_actions = policy.temporal_ensembler.ensembled_actions.clone()
    historical_ensembled_actions_count = policy.temporal_ensembler.ensembled_actions_count.clone()
    
    # Step 3: Sample n different actions at the current timestep
    samples = []
    
    for sample_idx in range(n_samples):
        # Restore the historical state before each sample
        policy.temporal_ensembler.ensembled_actions = historical_ensembled_actions.clone()
        policy.temporal_ensembler.ensembled_actions_count = historical_ensembled_actions_count.clone()
        
        # Now call select_action - this will:
        # 1. Generate a new chunk with a new z sample (stochastic VAE)
        # 2. Update the temporal ensemble with this new chunk
        # 3. Return the temporally ensembled action for current timestep
        action = policy.select_action(observation)
        samples.append(action.cpu().numpy())
    
    return samples


def sample_act_actions_direct_ensemble_access(
    policy, 
    observation, 
    n_samples: int,
    historical_chunks: List[torch.Tensor]  # Pre-computed historical chunks
) -> List[np.ndarray]:
    """
    Alternative approach: Directly manipulate the temporal ensemble computation
    without relying on the policy's internal state management.
    """
    
    # Prepare the observation batch
    batch = policy.normalize_inputs(observation)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = [batch[key] for key in policy.config.image_features]
    
    samples = []
    
    # Get the ensemble weights
    ensemble_weights = policy.temporal_ensembler.ensemble_weights
    
    for sample_idx in range(n_samples):
        # Generate a new chunk with new z sample
        with torch.no_grad():
            new_chunk = policy.model(batch)[0]  # New VAE sample each time
            new_chunk = policy.unnormalize_outputs({"action": new_chunk})["action"]
        
        # Manually compute the temporal ensemble
        # This combines historical_chunks + new_chunk according to the weighting scheme
        all_chunks = historical_chunks + [new_chunk]
        
        # Apply temporal ensemble weighting (simplified version)
        # In practice, you'd need to implement the full online averaging logic
        weighted_actions = []
        for i, chunk in enumerate(all_chunks):
            weight = ensemble_weights[i]
            weighted_actions.append(chunk * weight)
        
        # Sum and normalize
        ensembled_chunk = torch.stack(weighted_actions).sum(dim=0)
        ensembled_chunk = ensembled_chunk / ensemble_weights[:len(all_chunks)].sum()
        
        # Take the first action from the ensembled chunk
        action = ensembled_chunk[0, 0]  # [batch_idx=0, action_idx=0]
        samples.append(action.cpu().numpy())
    
    return samples


if __name__ == "__main__":
    print("Strategy 3: Proper temporal ensembling for heatmap sampling")
    print("This maintains FIFO buffer consistency while re-sampling z for each sample")