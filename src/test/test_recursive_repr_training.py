#!/usr/bin/env python3
"""
Diagnostic script to test recursive representation training and identify learning issues.
"""

import torch
import torch.nn as nn
import numpy as np
from pzero.pomuzero_model_mlp import POMuZeroModelMLP
import matplotlib.pyplot as plt
from typing import List, Tuple

def test_recursive_representation_learning():
    """Test the learning dynamics of recursive representation network."""
    
    print("🔍 Testing Recursive Representation Learning Dynamics...\n")
    
    # Configuration
    obs_shape = 27  # MiniGrid observation shape
    action_space_size = 7
    latent_state_dim = 128
    batch_size = 8
    sequence_length = 10
    
    # Create model
    model = POMuZeroModelMLP(
        observation_shape=obs_shape,
        action_space_size=action_space_size,
        latent_state_dim=latent_state_dim,
        use_recursive_representation=True,
        learned_initial_state=True,
        recursive_repr_hidden_channels=256,
        recursive_repr_layer_num=2,
    )
    
    # Create optimizer with gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Increased learning rate
    
    # Generate synthetic sequential data with more complex patterns
    def generate_synthetic_sequence(batch_size: int, seq_len: int, obs_shape: int) -> torch.Tensor:
        """Generate synthetic observation sequences with complex patterns to learn."""
        sequences = []
        for b in range(batch_size):
            # Create a sequence with multiple patterns (more challenging)
            base_obs = torch.randn(obs_shape) * 0.1
            sequence = []
            for t in range(seq_len):
                # Add multiple time-dependent patterns for better testing
                obs = base_obs.clone()
                obs += torch.sin(torch.tensor(t * 0.3)) * 0.3  # Slower sine wave
                obs += torch.cos(torch.tensor(t * 0.7)) * 0.2  # Faster cosine wave
                obs[t % obs_shape] += 0.5  # Position-dependent signal
                obs += torch.randn(obs_shape) * 0.03  # Reduced noise
                sequence.append(obs)
            sequences.append(torch.stack(sequence))
        return torch.stack(sequences)
    
    # Training loop
    losses = []
    gradient_norms = []
    latent_state_changes = []
    
    print("📚 Starting training loop...")
    
    for epoch in range(100):
        # Generate batch
        obs_sequences = generate_synthetic_sequence(batch_size, sequence_length, obs_shape)
        
        # Forward pass through sequence
        optimizer.zero_grad()
        
        # Process sequence
        total_loss = 0.0
        prev_latent_state = None
        latent_states = []
        
        for t in range(sequence_length):
            current_obs = obs_sequences[:, t]
            
            if t == 0:
                # Initial inference
                output = model.initial_inference(current_obs)
                latent_state = output.latent_state
            else:
                # Observation inference
                output = model.observation_inference(current_obs, prev_latent_state)
                latent_state = output.latent_state
            
            latent_states.append(latent_state)
            prev_latent_state = latent_state
            
            # Simple reconstruction loss (predict next observation)
            if t < sequence_length - 1:
                target_obs = obs_sequences[:, t + 1]
                # Use a simple linear layer to predict next observation from latent state
                if not hasattr(model, 'obs_predictor'):
                    model.obs_predictor = nn.Linear(latent_state_dim, obs_shape)
                
                predicted_obs = model.obs_predictor(latent_state)
                loss = nn.MSELoss()(predicted_obs, target_obs)
                total_loss += loss
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            
            # Add gradient clipping to test the fix
            torch.nn.utils.clip_grad_norm_(
                model.recursive_representation_network.parameters(), 
                max_norm=2.0  # Relaxed clipping
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # More relaxed overall clipping
            
            # Calculate gradient norm AFTER clipping
            total_norm = 0.0
            for p in model.recursive_representation_network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)
            
            # Calculate latent state change
            if len(latent_states) > 1:
                change = torch.norm(latent_states[-1] - latent_states[0]).item()
                latent_state_changes.append(change)
            
            optimizer.step()
            losses.append(total_loss.item())
        
        # Print progress
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.6f}, "
                  f"Grad Norm = {total_norm:.6f}, "
                  f"Latent Change = {latent_state_changes[-1] if latent_state_changes else 0:.6f}")
    
    # Analysis
    print("\n📊 Training Analysis:")
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Loss Change: {losses[0]:.6f} → {losses[-1]:.6f} ({(losses[-1]/losses[0] - 1)*100:.1f}%)")
    print(f"Average Gradient Norm: {np.mean(gradient_norms):.6f}")
    print(f"Average Latent State Change: {np.mean(latent_state_changes):.6f}")
    
    # Detect issues
    issues_found = []
    
    if abs(losses[-1] - losses[0]) / losses[0] < 0.01:
        issues_found.append("❌ Loss is not decreasing significantly")
    else:
        print("✅ Loss is decreasing properly")
    
    if np.mean(gradient_norms) < 1e-5:
        issues_found.append("❌ Gradients are too small (vanishing gradient problem)")
    elif np.mean(gradient_norms) > 10.0:
        issues_found.append("❌ Gradients are too large (exploding gradient problem)")
    else:
        print("✅ Gradient norms are in reasonable range")
    
    if np.mean(latent_state_changes) < 1e-4:
        issues_found.append("❌ Latent states are not changing enough")
    else:
        print("✅ Latent states are changing appropriately")
    
    if issues_found:
        print("\n⚠️  Issues Found:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected!")
    
    # Save plots
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        
        ax2.plot(gradient_norms)
        ax2.set_title('Gradient Norms')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_yscale('log')
        
        ax3.plot(latent_state_changes)
        ax3.set_title('Latent State Changes')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('L2 Change')
        
        # Latent state visualization
        with torch.no_grad():
            test_obs = generate_synthetic_sequence(1, sequence_length, obs_shape)
            latent_evolution = []
            prev_state = None
            
            for t in range(sequence_length):
                if t == 0:
                    output = model.initial_inference(test_obs[:, t])
                else:
                    output = model.observation_inference(test_obs[:, t], prev_state)
                latent_evolution.append(output.latent_state[0, :10].numpy())  # First 10 dims
                prev_state = output.latent_state
            
            latent_evolution = np.array(latent_evolution)
            im = ax4.imshow(latent_evolution.T, aspect='auto', cmap='viridis')
            ax4.set_title('Latent State Evolution')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Latent Dimension')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('recursive_repr_training_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Training analysis saved to 'recursive_repr_training_analysis.png'")
        
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}")
    
    return issues_found

def test_gradient_flow():
    """Test gradient flow through recursive representation network."""
    print("\n🔍 Testing Gradient Flow...\n")
    
    model = POMuZeroModelMLP(
        observation_shape=27,
        action_space_size=7,
        latent_state_dim=128,
        use_recursive_representation=True,
        learned_initial_state=True,
    )
    
    # Create dummy data
    obs = torch.randn(4, 27, requires_grad=True)
    prev_latent = torch.randn(4, 128, requires_grad=True)
    
    # Forward pass
    output = model.observation_inference(obs, prev_latent)
    loss = output.latent_state.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("Gradient analysis:")
    for name, param in model.recursive_representation_network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"  {name}: No gradient!")
    
    # Check input gradients
    if obs.grad is not None:
        print(f"  obs gradient norm: {obs.grad.norm().item():.6f}")
    if prev_latent.grad is not None:
        print(f"  prev_latent gradient norm: {prev_latent.grad.norm().item():.6f}")

if __name__ == "__main__":
    # Run diagnostics
    test_gradient_flow()
    issues = test_recursive_representation_learning()
    
    if not issues:
        print("\n🎉 Recursive representation appears to be working correctly!")
        print("   If you're still not seeing learning, check:")
        print("   1. Episode-based data sampling in your training loop")
        print("   2. Observation sequence preparation")
        print("   3. Environment reward signals")
    else:
        print("\n🔧 Please address the issues above to improve learning.") 