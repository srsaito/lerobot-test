"""
Supporting both temporal ensembling and non-temporal ensembling modes in the visualizer
"""

class PolicyVisualizationWrapper:
    """
    Wrapper that automatically detects and handles both ACT modes:
    1. Regular ACT (temporal_ensemble_coeff=null, n_action_steps=100)
    2. Temporal Ensembling ACT (temporal_ensemble_coeff=0.01, n_action_steps=1)
    """
    
    def __init__(self, policy):
        self.policy = policy
        self._visualization_cache = None
        
        # Detect which mode we're in
        self.is_temporal_ensembling = self._detect_temporal_ensembling_mode()
        
    def _detect_temporal_ensembling_mode(self):
        """
        Detect if policy is configured for temporal ensembling
        """
        config = self.policy.config
        
        # Temporal ensembling is enabled if:
        # 1. temporal_ensemble_coeff is not None, AND
        # 2. n_action_steps is 1
        is_te_enabled = (
            config.temporal_ensemble_coeff is not None and 
            config.n_action_steps == 1
        )
        
        print(f"Detected mode: {'Temporal Ensembling' if is_te_enabled else 'Regular Action Chunking'}")
        print(f"  temporal_ensemble_coeff: {config.temporal_ensemble_coeff}")
        print(f"  n_action_steps: {config.n_action_steps}")
        
        return is_te_enabled
    
    def process_timestep(self, observation):
        """
        Process one timestep with mode-specific logic
        """
        # Clear visualization cache
        self._visualization_cache = None
        
        # 1. Get main action (works the same for both modes)
        main_action = self.policy.select_action(observation)
        
        # 2. Cache observation for visualization
        self._visualization_cache = self._encode_observation(observation)
        
        # 3. Generate heatmap samples (mode-specific logic)
        if self.is_temporal_ensembling:
            heatmap_samples = self._sample_heatmap_temporal_ensembling()
        else:
            heatmap_samples = self._sample_heatmap_regular_chunking()
        
        # 4. Generate forward trace (mode-specific logic)
        if self.is_temporal_ensembling:
            forward_trace = self._predict_forward_trace_temporal_ensembling()
        else:
            forward_trace = self._predict_forward_trace_regular_chunking()
        
        return main_action, heatmap_samples, forward_trace
    
    def _sample_heatmap_temporal_ensembling(self):
        """
        Heatmap sampling for temporal ensembling mode
        
        Key insight: Must preserve FIFO buffer state while re-sampling current chunk
        """
        # Save current FIFO buffer state
        historical_state = self._save_temporal_ensemble_state()
        
        samples = []
        for _ in range(self.n_samples):
            # Restore historical state
            self._restore_temporal_ensemble_state(historical_state)
            
            # Generate new chunk with different z sample
            new_chunk = self._generate_chunk_with_cached_obs()
            
            # Update FIFO and get ensembled action
            ensembled_action = self.policy.temporal_ensembler.update(new_chunk)
            samples.append(ensembled_action.cpu().numpy())
        
        return samples
    
    def _sample_heatmap_regular_chunking(self):
        """
        Heatmap sampling for regular action chunking mode
        
        Key insight: No FIFO buffer to preserve, just sample new chunks
        """
        samples = []
        for _ in range(self.n_samples):
            # Generate new action chunk with different z sample
            actions = self.policy.model(self._visualization_cache)[0]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
            
            # Take first action from chunk
            first_action = actions[0, 0]  # [batch_idx=0, action_idx=0]
            samples.append(first_action.cpu().numpy())
        
        return samples
    
    def _predict_forward_trace_temporal_ensembling(self):
        """
        Forward trace for temporal ensembling mode
        
        Challenge: How to predict future temporally ensembled actions?
        We need to simulate the FIFO buffer evolution over time.
        """
        # This is complex - we need to simulate multiple timesteps
        # Each future timestep would have its own FIFO buffer state
        
        # Placeholder for now
        return self._simulate_temporal_ensemble_forward_trace()
    
    def _predict_forward_trace_regular_chunking(self):
        """
        Forward trace for regular chunking mode
        
        Simple: Generate one chunk and take the next n_forward actions
        """
        # Generate one action chunk
        actions = self.policy.model(self._visualization_cache)[0]
        actions = self.policy.unnormalize_outputs({"action": actions})["action"]
        
        # Take next n_forward actions from the chunk
        forward_actions = actions[0, 1:self.n_forward+1]  # Skip first action (current)
        
        return forward_actions.cpu().numpy()
    
    def _save_temporal_ensemble_state(self):
        """Save current temporal ensembler state"""
        if hasattr(self.policy, 'temporal_ensembler'):
            return {
                'ensembled_actions': self.policy.temporal_ensembler.ensembled_actions.clone() if self.policy.temporal_ensembler.ensembled_actions is not None else None,
                'ensembled_actions_count': self.policy.temporal_ensembler.ensembled_actions_count.clone() if self.policy.temporal_ensembler.ensembled_actions_count is not None else None
            }
        return None
    
    def _restore_temporal_ensemble_state(self, saved_state):
        """Restore temporal ensembler state"""
        if saved_state and hasattr(self.policy, 'temporal_ensembler'):
            self.policy.temporal_ensembler.ensembled_actions = saved_state['ensembled_actions'].clone() if saved_state['ensembled_actions'] is not None else None
            self.policy.temporal_ensembler.ensembled_actions_count = saved_state['ensembled_actions_count'].clone() if saved_state['ensembled_actions_count'] is not None else None

# Usage example
def example_usage():
    """
    How the visualizer would work with both modes
    """
    
    # Load policy (could be either mode based on config.json)
    policy = ACTPolicy.from_pretrained("ssaito/act_pusht_test")
    
    # Create wrapper (automatically detects mode)
    viz_wrapper = PolicyVisualizationWrapper(policy)
    
    # Use the same interface regardless of mode
    for t in range(time_steps):
        observation = env.get_observation()
        
        # This works for both temporal ensembling and regular chunking
        main_action, heatmap_samples, forward_trace = viz_wrapper.process_timestep(observation)
        
        # Visualization code remains the same
        plot_heatmap(heatmap_samples)
        plot_forward_trace(forward_trace)
        
        # Execute action in environment
        env.step(main_action)

# Key insight: The complexity is hidden inside the wrapper
def design_principle():
    """
    Design principle: The visualizer user doesn't need to know which mode is active.
    The wrapper automatically detects and handles the differences internally.
    
    User just needs to:
    1. Edit config.json to enable/disable temporal ensembling
    2. Run the same visualization code
    3. Get appropriate behavior for each mode
    """
    pass