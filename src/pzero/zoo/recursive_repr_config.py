"""
Configuration helper for Recursive Representation in POMuZero

This module provides configuration templates and helper functions for setting up
recursive representation networks in POMuZero with episode-based sampling.
"""

import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

def get_recursive_repr_config_template():
    """
    Returns a configuration template for recursive representation network.
    Users should merge this with their existing POMuZero config.
    """
    
    recursive_config = {
        # Model configuration for recursive representation
        'model': {
            'use_recursive_representation': True,
            'recursive_repr_hidden_channels': 256,
            'recursive_repr_layer_num': 2,
            'res_connection_in_recursive_repr': False,
            'learned_initial_state': True,  # Use learned initial state vs zero initialization
        },
        
        # Buffer configuration for episode sampling
        'replay_buffer': {
            'sample_type': 'episode',  # Use 'episode' instead of 'transition'
            'replay_buffer_size': 1000,  # Adjust based on episode length
            'game_segment_length': 200,  # Length of game segments (episodes)
        },
        
        # Training configuration adjustments
        'batch_size': 32,  # May need smaller batch size for episode sampling
        'num_unroll_steps': 5,  # Keep standard unroll steps
        
        # Additional recommendations
        'use_priority': True,  # Priority sampling still beneficial
        'reanalyze_ratio': 1.0,  # Full reanalysis for recursive representation
    }
    
    return recursive_config

def get_default_pomuzero_config():
    """
    Returns a complete default configuration for POMuZero with recursive representation.
    This can be used as a starting point for new experiments.
    """
    
    config = {
        'type': 'pomuzero',
        'cuda': True,
        'on_policy': False,
        'priority': True,
        'priority_prob_alpha': 0.6,
        'priority_prob_beta': 0.4,
        'nstep': 5,
        'discount_factor': 0.997,
        
        # Episode-based sampling
        'replay_buffer_size': 1000,
        'game_segment_length': 200,
        'sample_type': 'episode',
        
        # Training
        'batch_size': 32,
        'learning_rate': 0.003,
        'num_unroll_steps': 5,
        'td_steps': 5,
        'update_per_collect': 50,
        'reanalyze_ratio': 1.0,
        
        # MCTS
        'mcts_ctree': True,
        'num_simulations': 50,
        'root_dirichlet_alpha': 0.3,
        'root_noise_weight': 0.25,
        
        # Model with recursive representation
        'model': {
            'observation_shape': 8,  # Adjust for your environment
            'action_space_size': 4,   # Adjust for your environment
            'model_type': 'mlp',
            'latent_state_dim': 64,
            
            # Recursive representation settings
            'use_recursive_representation': True,
            'recursive_repr_hidden_channels': 256,
            'recursive_repr_layer_num': 2,
            'res_connection_in_recursive_repr': False,
            'learned_initial_state': True,
            
            # Standard settings
            'support_scale': 300,
            'reward_support_size': 601,
            'value_support_size': 601,
            'categorical_distribution': True,
        },
    }
    
    return config

def validate_recursive_repr_config(config):
    """
    Validates that the configuration is compatible with recursive representation.
    
    Args:
        config (dict): The configuration dictionary to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check if recursive representation is enabled
    if not config.get('model', {}).get('use_recursive_representation', False):
        errors.append("use_recursive_representation must be True for recursive representation")
    
    # Check sampling type
    sample_type = config.get('sample_type', config.get('replay_buffer', {}).get('sample_type', 'transition'))
    if sample_type != 'episode':
        errors.append("sample_type should be 'episode' for optimal recursive representation performance")
    
    # Check batch size (recommend smaller for episode sampling)
    batch_size = config.get('batch_size', 64)
    if batch_size > 64:
        errors.append(f"batch_size ({batch_size}) is large for episode sampling, consider reducing to 32-64")
    
    # Check game segment length
    game_segment_length = config.get('replay_buffer', {}).get('game_segment_length', 200)
    if game_segment_length < 50:
        errors.append(f"game_segment_length ({game_segment_length}) may be too short for recursive representation")
    
    # Check reanalyze ratio (should be high for recursive representation)
    reanalyze_ratio = config.get('reanalyze_ratio', 0.0)
    if reanalyze_ratio < 0.8:
        errors.append(f"reanalyze_ratio ({reanalyze_ratio}) should be high (≥0.8) for recursive representation")
    
    return len(errors) == 0, errors

def print_config_recommendations():
    """Print recommendations for configuring recursive representation."""
    
    logger.info("=" * 60)
    logger.info("RECURSIVE REPRESENTATION CONFIGURATION GUIDE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Key Configuration Changes:")
    logger.info("1. Model: Set use_recursive_representation=True")
    logger.info("2. Buffer: Set sample_type='episode'")
    logger.info("3. Training: Consider smaller batch_size (32-64)")
    logger.info("4. Reanalysis: Set reanalyze_ratio=1.0")
    logger.info("")
    logger.info("Example usage:")
    logger.info("```python")
    logger.info("from recursive_repr_config import get_recursive_repr_config_template")
    logger.info("config = get_recursive_repr_config_template()")
    logger.info("# Merge with your existing config or use as base")
    logger.info("```")
    logger.info("")
    logger.info("For complete config:")
    logger.info("```python")
    logger.info("from recursive_repr_config import get_default_pomuzero_config")
    logger.info("config = get_default_pomuzero_config()")
    logger.info("```")
    logger.info("=" * 60)

if __name__ == "__main__":
    print_config_recommendations()
    
    # Example validation
    config = get_default_pomuzero_config()
    is_valid, errors = validate_recursive_repr_config(config)
    
    logger.info("\nValidation Result:")
    if is_valid:
        logger.info("✅ Configuration is valid for recursive representation")
    else:
        logger.warning("❌ Configuration issues found:")
        for error in errors:
            logger.warning(f"  - {error}") 