"""
Analyzing different approaches for using ACT with temporal ensembling
"""

# Scenario 1: Using existing ACT policy with temporal ensembling
def scenario_1_modify_existing_act():
    """
    Using your existing HuggingFace model (registered as "act") with temporal ensembling
    """
    
    # Your existing model on HF Hub
    model_id = "ssaito/act_pusht_test"  # Saved as policy_type="act"
    
    # Option 1A: Load and modify config programmatically
    config = ACTConfig.from_pretrained(model_id)
    config.temporal_ensemble_coeff = 0.01  # Enable temporal ensembling
    config.n_action_steps = 1              # Required for temporal ensembling
    policy = ACTPolicy(config)
    policy.load_state_dict_from_pretrained(model_id)
    
    # Option 1B: CLI overrides
    # python control_robot.py --policy.type=act --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
    
    return "Requires manual config modification each time"

# Scenario 2: Separate act-te policy type
def scenario_2_separate_act_te():
    """
    Creating a separate "act-te" policy type for temporal ensembling
    """
    
    # New config class with temporal ensembling defaults
    @PreTrainedConfig.register_subclass("act-te")
    class ACTTemporalEnsembleConfig(ACTConfig):
        temporal_ensemble_coeff: float = 0.01  # Default enabled
        n_action_steps: int = 1                # Default required value
    
    # New policy class
    class ACTTemporalEnsemblePolicy(ACTPolicy):
        config_class = ACTTemporalEnsembleConfig
        name = "act-te"
    
    # Usage with existing model weights
    config = ACTTemporalEnsembleConfig.from_pretrained("ssaito/act_pusht_test")
    policy = ACTTemporalEnsemblePolicy(config)
    
    # The key question: Can we load "act" weights into "act-te" policy?
    return "Clean interface, but weight compatibility question"

# The mapping challenge
def weight_compatibility_analysis():
    """
    Can we load weights from "act" policy into "act-te" policy?
    
    Key insight: The underlying neural network (ACT transformer) is identical.
    Only the inference behavior changes (temporal ensembling vs action queue).
    
    So weight loading should work fine - it's just the config that differs.
    """
    
    # This should work:
    act_te_policy = ACTTemporalEnsemblePolicy.from_pretrained("ssaito/act_pusht_test")
    
    # Because:
    # 1. Same underlying ACT transformer weights
    # 2. Same input/output dimensions  
    # 3. Only inference logic differs (temporal ensembler vs action queue)
    
    return "Weight compatibility should not be an issue"