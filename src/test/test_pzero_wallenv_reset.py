#!/usr/bin/env python3

import sys
import os
from easydict import EasyDict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pzero.zoo.minigrid_pzero_env import MiniGridEnvPZero
from minigrid.core.actions import Actions

STEP_COUNT = 300

def setup_resetting_wallenv():
    """Setup MiniGridEnvPZero with MiniGrid-WallEnvReset-5x5-v0"""
    
    # Create configuration for the environment
    config = EasyDict({
        'env_id': 'MiniGrid-WallEnvReset-5x5-v0',
        'view_size': 3,
        'max_step': STEP_COUNT,
        'save_replay_gif': True,
        'replay_path_gif': './src/test/gifs/'
    })
    
    # Initialize the environment
    env = MiniGridEnvPZero(config)
    
    # Set seed for reproducible results
    env.seed(seed=42, dynamic_seed=False)
    
    print(f"Testing environment: {config['env_id']}")
    print(f"Max steps: {config['max_step']}")
    

    # Reset the environment
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    return env
    

def test_random_policy(env):
    # Run random policy until termination
    step_count = 0
    total_reward = 0
    done = False
    
    print("\nRunning random policy...")
    
    while not done:
        # Get random action
        action = env.random_action()
        
        # Take step
        timestep = env.step(action)
        obs = timestep.obs
        reward = timestep.reward
        done = timestep.done
        info = timestep.info
        
        total_reward += reward
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"Step {step_count}: reward={reward}, total_reward={total_reward}")
    
    print(f"\nEpisode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"Final info: {info}")
    
    # # Test multiple episodes
    # print("\nTesting multiple episodes...")
    # for episode in range(3):
    #     obs = env.reset()
    #     episode_reward = 0
    #     episode_steps = 0
    #     done = False
        
    #     while not done:
    #         action = env.random_action()
    #         timestep = env.step(action)
    #         episode_reward += timestep.reward
    #         episode_steps += 1
    #         done = timestep.done
        
         #     print(f"Episode {episode + 1}: {episode_steps} steps, reward: {episode_reward}")


def test_actions_to_goal(env):
    """Test specific action sequence to the goal"""
    
    # Reset environment for clean start
    obs = env.reset()
    
    action_sequence = [
        # Actions.forward,  
        # Actions.right, 
        Actions.left,      
        Actions.forward,  
        Actions.forward,   
        Actions.right,     
        Actions.forward   
    ]
    
    action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    sequence_names = [action_names[action] for action in action_sequence]
    
    print(f"Executing action sequence: {' -> '.join(sequence_names)}")
    print(f"Action indices: {action_sequence}")
    
    step_count = 0
    total_reward = 0
    done = False
    
    # Execute the planned sequence
    for i, action in enumerate(action_sequence):
        if done:
            print(f"Episode ended before completing sequence at step {i}")
            break
            
        print(f"Step {step_count + 1}: Executing {action_names[action]} (action {action})")
        
        # Take the action
        timestep = env.step(action)
        obs = timestep.obs
        reward = timestep.reward
        done = timestep.done
        info = timestep.info
        
        total_reward += reward
        step_count += 1
        
        print(f"  Reward: {reward}, Done: {done}")
        if done:
            print(f"Episode completed during planned sequence!")
            print(f"Total steps: {step_count}")
            print(f"Total reward: {total_reward}")
            print(f"Final info: {info}")
            return
    
    print(f"\nCompleted planned sequence. Now playing random actions...")
    
    # Continue with random actions until done
    random_step_count = 0
    while not done:
        action = env.random_action()
        
        timestep = env.step(action)
        obs = timestep.obs
        reward = timestep.reward
        done = timestep.done
        info = timestep.info
        
        total_reward += reward
        step_count += 1
        random_step_count += 1
        
        if random_step_count % 20 == 0:
            print(f"Random step {random_step_count} (total step {step_count}): reward={reward}")
    
    print(f"\nEpisode completed!")
    print(f"Planned sequence steps: {len(action_sequence)}")
    print(f"Random action steps: {random_step_count}")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"Final info: {info}")

    assert step_count == STEP_COUNT
    assert total_reward >= (STEP_COUNT * -0.01)+1


if __name__ == "__main__":
    env = setup_resetting_wallenv()
    # test_random_policy(env)
    test_actions_to_goal(env)
    