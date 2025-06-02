import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from easydict import EasyDict
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from zoo.minigrid.envs.minigrid_lightzero_env import MiniGridEnvLightZero
from lzero.policy.muzero import MuZeroPolicy
import imageio
from zoo.minigrid.config.minigrid_muzero_config import main_config, create_config

def display_frames_as_gif(frames: list, path: str) -> None:
    """
    Convert a list of frames into a GIF and save it using imageio.
    Args:
        frames (list): List of frames to be saved as a GIF
        path (str): Path where the GIF will be saved
    """
    imageio.mimsave(path, frames, fps=20)

def run_episode(env, policy, seed=None):
    """
    Run a single episode with the given policy and environment.
    """
    if seed is not None:
        env.seed(seed)
    
    obs = env.reset()
    frames = []
    done = False
    episode_return = 0
    
    while not done:
        # Get action from policy
        action = policy.forward(obs)
        
        # Step environment
        timestep = env.step(action)
        obs = timestep.obs
        done = timestep.done
        episode_return += timestep.reward
        
        # Save frame
        frame = env.render()
        frames.append(frame)
    
    return frames, episode_return

def main():
    # Use the configuration from minigrid_eval.py
    env_config = main_config.env
    policy_config = main_config.policy
    
    # Enable saving replay as GIF
    env_config.save_replay_gif = True
    env_config.replay_path_gif = './episode_gifs'
    
    # Create environment
    env = MiniGridEnvLightZero(env_config)
    
    # Create and load policy
    policy = MuZeroPolicy(policy_config)
    
    # Load the trained model if path is provided
    model_path = './ckpt/ckpt_best.pth.tar'  # Update this path to your model checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(checkpoint['model'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        return
    
    policy.eval()
    
    # Create directory for saving GIFs
    gif_dir = './episode_gifs'
    os.makedirs(gif_dir, exist_ok=True)
    
    # Run episode and save GIF
    frames, episode_return = run_episode(env, policy, seed=42)
    
    # Save GIF
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    gif_path = os.path.join(gif_dir, f'episode_return_{episode_return:.2f}_{timestamp}.gif')
    display_frames_as_gif(frames, gif_path)
    print(f"Saved episode GIF to {gif_path}")
    print(f"Episode return: {episode_return:.2f}")

if __name__ == "__main__":
    main() 