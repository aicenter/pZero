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
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Force reconfiguration even if already configured
    )
    
    # Explicitly set level for all relevant loggers
    loggers_to_configure = [
        'pzero',
        'pzero.pomuzero', 
        'pzero.zoo',
        'src.pzero',
        'src.pzero.pomuzero',
        'src.pzero.zoo'
    ]
    
    for logger_name in loggers_to_configure:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(numeric_level)
    
    # Get the script logger
    script_logger = logging.getLogger(__name__)
    
    # Test debug logging is working
    script_logger.debug(f"Debug logging is enabled at level: {log_level}")
    
    return script_logger


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
    
    # Setup logging based on CLI argument BEFORE importing pzero modules
    logger = setup_logging(args.log_level)
    
    logger.info(f"Starting POMuZero training with environment: {args.env_id}")
    logger.info(f"Log level set to: {args.log_level}")
    logger.debug("This is a test debug message from the script")
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Import after logging is configured
    logger.debug("Importing pzero modules...")
    from pzero.train_pomuzero import train_pomuzero
    from pzero.zoo.pomuzero_recursive_minigrid_config import create_config # creates config for the environment

    logger.debug("Creating configuration...")
    main_config, create_config, seed, max_env_step = create_config(env_id=args.env_id)
    logger.debug("Starting training...")
    train_pomuzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
