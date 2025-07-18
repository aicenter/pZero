#!/usr/bin/env python3
"""
Script to run POMuZero training on MiniGrid environments with configurable env_id.
"""

import argparse
import logging
import sys
import os
from easydict import EasyDict


def setup_logging(log_level):
    """Configure logging based on the specified log level."""
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run POMuZero training on MiniGrid environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment configuration
    parser.add_argument('--env-id', type=str, default='MiniGrid-WallEnvReset-5x5-v0',
                       help='MiniGrid environment ID to use for training')
    
    # Logging configuration
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    
    args = parser.parse_args()
    
    # Setup logging based on CLI argument
    logger = setup_logging(args.log_level)
    
    logger.info(f"Starting POMuZero training with environment: {args.env_id}")
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    from pzero.train_pomuzero import train_pomuzero
    from pzero.zoo.pomuzero_minigrid_config import create_config # creates config for the environment

    main_config, create_config, seed, max_env_step = create_config(env_id=args.env_id)
    train_pomuzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
