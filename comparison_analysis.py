"""
Detailed comparison of PolicyVisualizationWrapper vs PolicyAnalyzer approaches
"""

# Example implementations to illustrate the differences

# Option A: PolicyVisualizationWrapper
class PolicyVisualizationWrapper:
    def __init__(self, policy):
        self.policy = policy
        # Can cache expensive computations
        self._cached_historical_chunks = None
        self._cached_observation_features = None
        
    def sample_for_heatmap(self, observation, n_samples):
        # Can reuse cached state between calls
        if self._cached_observation_features is None:
            self._cached_observation_features = self._extract_features(observation)
        return self._sample_with_cache(n_samples)
    
    def predict_forward_trace(self, observation, n_forward):
        # Can build on previously cached state
        return self._predict_using_cache(n_forward)

# Option C: PolicyAnalyzer  
class PolicyAnalyzer:
    @staticmethod
    def sample_for_heatmap(policy, observation, n_samples):
        # Must recompute everything each call
        features = PolicyAnalyzer._extract_features(policy, observation)
        return PolicyAnalyzer._sample_fresh(policy, features, n_samples)
    
    @staticmethod
    def predict_forward_trace(policy, observation, n_forward):
        # Must recompute features again
        features = PolicyAnalyzer._extract_features(policy, observation)
        return PolicyAnalyzer._predict_fresh(policy, features, n_forward)