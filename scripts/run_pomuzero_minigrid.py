#!/usr/bin/env python3
"""
Script to run POMuZero training on MiniGrid environments with configurable env_id.
"""

import argparse
import sys
import os
from easydict import EasyDict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run POMuZero training on MiniGrid environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment configuration
    parser.add_argument('--env-id', type=str, default='MiniGrid-WallEnvReset-5x5-v0',
                       help='MiniGrid environment ID to use for training')
    
    
    args = parser.parse_args()
    
    print(f"Starting POMuZero training with environment: {args.env_id}")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")


    from pzero.train_pomuzero import train_pomuzero
    from pzero.zoo.pomuzero_minigrid_config import create_config # creates config for the environment

    main_config, create_config, seed, max_env_step = create_config(env_id=args.env_id)
    train_pomuzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
