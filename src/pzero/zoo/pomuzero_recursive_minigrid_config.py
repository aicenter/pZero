from easydict import EasyDict
import logging
import pzero.zoo.wallenv
# from pzero.zoo.minigrid_pzero_env import MiniGridEnvLightZero
from pzero.zoo.recursive_repr_config import validate_recursive_repr_config

# Initialize logger for this module
logger = logging.getLogger(__name__)

# The typical MiniGrid env id: {'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0', 'MiniGrid-DoorKey-8x8-v0','MiniGrid-DoorKey-16x16-v0'},
# please refer to https://github.com/Farama-Foundation/MiniGrid for details.

# ===================================================================
# RECURSIVE REPRESENTATION POMUZERO CONFIG FOR MINIGRID
# ===================================================================
# This configuration uses recursive representation networks where:
# - Representation network takes (observation, prev_latent_state) → new_latent_state
# - Uses episode-based sampling instead of transition-based
# - Optimized hyperparameters for recursive representation training
# ===================================================================

# env_id = 'MiniGrid-DoorKey-5x5-v0'
# env_id = 'MiniGrid-WallEnv-5x5-v0'

def create_config(env_id='MiniGrid-WallEnvReset-5x5-v0', seed=0, max_env_step=int(1e6)):
    # env_id = 'MiniGrid-WallEnvReset-5x5-v0'
    # max_env_step = int(1e6)

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    # seed = 0
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50 # Number of MCTS simulations
    update_per_collect = 200 # Number of updates on NNs for each batch of data
    batch_size = 64 # Reduced batch size for episode-based sampling with recursive representation
    reanalyze_ratio = 1.0 # Full reanalysis recommended for recursive representation
    td_steps = 5 # how many steps to unroll to calculate the target value of a position
    policy_entropy_weight = 0.01  # Increased exploration for better learning signal
    threshold_training_steps_for_final_temperature = int(5e5)
    eps_greedy_exploration_in_collect = False
    
    # Recursive representation settings
    use_recursive_representation = True
    recursive_repr_hidden_channels = 256
    recursive_repr_layer_num = 2
    learned_initial_state = True
    sample_type = 'episode'  # Episode-based sampling for recursive representation
    
    # Improved hyperparameters for recursive representation
    learning_rate = 0.001  # Increased from 0.0005 to 0.001 for better learning with stable gradients
    grad_clip_value = 2.0   # Relaxed gradient clipping 
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    # Common base path for experiment data
    base_data_path = 'data/pomuzero_recursive_view3x3_reset'
    exp_name = f'{base_data_path}/{env_id}_pomuzero_recursive_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_' \
            f'collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}_pelw{policy_entropy_weight}_seed{seed}'

    minigrid_muzero_config = dict(
        exp_name=exp_name,
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            continuous=False,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # (bool) If True, save the replay as a gif file.
            save_replay_gif=True,
            # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
            replay_path_gif=f'./{exp_name}/gif/',
            # reduce max steps for environment without terminal state
            max_step=150,
        ),
        policy=dict(
            model=dict(
                observation_shape=27, # should be a number of values in the flattened observation. 
                action_space_size=7,
                model_type='po_mlp',
                lstm_hidden_size=256,
                latent_state_dim=512,
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
                self_supervised_learning_loss=True,  # NOTE: default is False.
                
                # Recursive representation settings
                use_recursive_representation=use_recursive_representation,
                recursive_repr_hidden_channels=recursive_repr_hidden_channels,
                recursive_repr_layer_num=recursive_repr_layer_num,
                res_connection_in_recursive_repr=False,
                learned_initial_state=learned_initial_state,
            ),
            eps=dict(
                eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
                decay=int(2e5),
            ),
            policy_entropy_weight=policy_entropy_weight,
            td_steps=td_steps,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
            cuda=True,
            env_type='not_board_games',
            game_segment_length=200,  # Increased for better episode sampling
            sample_type=sample_type,  # Episode-based sampling for recursive representation
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
            ssl_loss_weight=2,  # NOTE: default is 0.
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e2),
            replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    )

    minigrid_muzero_config = EasyDict(minigrid_muzero_config)
    main_config = minigrid_muzero_config

    # Validate recursive representation configuration
    config_dict = dict(minigrid_muzero_config.policy)
    is_valid, errors = validate_recursive_repr_config(config_dict)
    
    if not is_valid:
        logger.warning("⚠️  Configuration validation warnings:")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✅ Recursive representation configuration is valid!")
    
    # Print configuration summary
    logger.info(f"\n🔧 Recursive Representation Config Summary:")
    logger.info(f"  - Use Recursive Representation: {use_recursive_representation}")
    logger.info(f"  - Sample Type: {sample_type}")
    logger.info(f"  - Batch Size: {batch_size} (optimized for episode sampling)")
    logger.info(f"  - Reanalyze Ratio: {reanalyze_ratio}")
    logger.info(f"  - Game Segment Length: {main_config.policy.game_segment_length}")
    logger.info(f"  - Recursive Hidden Channels: {recursive_repr_hidden_channels}")
    logger.info(f"  - Learned Initial State: {learned_initial_state}")

    minigrid_muzero_create_config = dict(
        env=dict(
            type='minigrid_pzero',
            import_names=['pzero.zoo.minigrid_pzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='pomuzero',
            import_names=['pzero.pomuzero'],
        ),
        collector=dict(
            type='episode_muzero',
            import_names=['lzero.worker.muzero_collector'],
        )
    )
    minigrid_muzero_create_config = EasyDict(minigrid_muzero_create_config)
    create_config = minigrid_muzero_create_config
    return main_config, create_config, seed, max_env_step

if __name__ == "__main__":
    from pzero.train_pomuzero import train_pomuzero
    main_config, create_config, seed, max_env_step = create_config()
    train_pomuzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
