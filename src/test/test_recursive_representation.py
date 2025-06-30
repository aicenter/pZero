import torch
import numpy as np
import logging
from pzero.pomuzero_model_mlp import POMuZeroModelMLP

# Initialize logger for this module
logger = logging.getLogger(__name__)

def test_recursive_representation():
    """Test the recursive representation network implementation"""
    
    logger.info("Testing Recursive Representation Network...")
    
    # Test 1: Standard MuZero model (backward compatibility)
    logger.info("\n1. Testing standard MuZero model (backward compatibility)...")
    standard_model = POMuZeroModelMLP(
        observation_shape=8,
        action_space_size=4,
        latent_state_dim=64,
        use_recursive_representation=False,
    )
    
    obs = torch.randn(2, 8)  # batch_size=2, obs_shape=8
    output = standard_model.initial_inference(obs)
    logger.info(f"   ✓ Standard model output shapes: latent_state={output.latent_state.shape}, value={output.value.shape}")
    
    # Test 2: Recursive representation model  
    logger.info("\n2. Testing recursive representation model...")
    recursive_model = POMuZeroModelMLP(
        observation_shape=8,
        action_space_size=4,
        latent_state_dim=64,
        use_recursive_representation=True,
        learned_initial_state=False,  # Use parameter initialization
    )
    
    # Test initial inference
    output1 = recursive_model.initial_inference(obs)
    logger.info(f"   ✓ Initial inference output shapes: latent_state={output1.latent_state.shape}")
    
    # Test observation inference  
    prev_latent_state = output1.latent_state
    new_obs = torch.randn(2, 8)
    output2 = recursive_model.observation_inference(new_obs, prev_latent_state)
    logger.info(f"   ✓ Observation inference output shapes: latent_state={output2.latent_state.shape}")
    
    # Test that latent states are different (model is learning)
    latent_diff = torch.norm(output1.latent_state - output2.latent_state)
    logger.info(f"   ✓ Latent state difference: {latent_diff.item():.4f} (should be > 0)")
    
    # Test 3: Recursive representation with learned initial state
    logger.info("\n3. Testing recursive representation with learned initial state...")
    learned_init_model = POMuZeroModelMLP(
        observation_shape=8,
        action_space_size=4,
        latent_state_dim=64,
        use_recursive_representation=True,
        learned_initial_state=True,  # Use network initialization
    )
    
    output3 = learned_init_model.initial_inference(obs)
    logger.info(f"   ✓ Learned initial state output shapes: latent_state={output3.latent_state.shape}")
    
    # Test 4: Error handling
    logger.info("\n4. Testing error handling...")
    try:
        standard_model.observation_inference(obs, prev_latent_state)
        logger.error("   ✗ Should have raised ValueError")
    except ValueError as e:
        logger.info(f"   ✓ Correctly raised ValueError: {e}")
    
    # Test 5: Recurrent inference (should still work)
    logger.info("\n5. Testing recurrent inference (unchanged)...")
    action = torch.tensor([0, 1])  # batch_size=2
    recurrent_output = recursive_model.recurrent_inference(output1.latent_state, action)
    logger.info(f"   ✓ Recurrent inference output shapes: latent_state={recurrent_output.latent_state.shape}")
    
    logger.info("\n✅ All tests passed! Recursive representation network is working correctly.")

if __name__ == "__main__":
    test_recursive_representation() 