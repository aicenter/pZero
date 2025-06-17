import copy
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.minigrid.envs.minigrid_wrapper import ViewSizeWrapper
from dizoo.minigrid.envs.minigrid_env import MiniGridEnv
from easydict import EasyDict
from matplotlib import animation
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

from PIL import Image, ImageDraw, ImageFont

@ENV_REGISTRY.register('minigrid_pzero')
class MiniGridEnvPZero(MiniGridEnv):
    """
    Overview:
        A MiniGrid environment for LightZero, based on OpenAI Gym.
    Attributes:
        config (dict): Configuration dict. Default configurations can be updated using this.
        _cfg (dict): Internal configuration dict that stores runtime configurations.
        _init_flag (bool): Flag to check if the environment is initialized.
        _env_id (str): The name of the MiniGrid environment.
        # _flat_obs (bool): Flag to check if flat observations are returned.
        _save_replay (bool): Flag to check if replays are saved.
        _max_step (int): Maximum number of steps for the environment.
    """
    config = dict(
        # (str) The gym environment name.
        env_id='MiniGrid-Empty-8x8-v0',
        # (bool) Whether to use flat observation.
        # _flat_obs=True,
        # (bool) If True, save the replay as a gif file.
        save_replay_gif=False,
        # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
        replay_path_gif=None,
        view_size=3,
        # flat_obs=True,
        # (int) The maximum number of steps for each episode.
        max_step=300,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Returns the default configuration with the current environment class name.
        Returns:
            - cfg (:obj:`dict`): Configuration dict.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The configuration should include the environment name,
                        whether to use flat observations, and the maximum number of steps.
        """
        self._cfg = cfg
        self._init_flag = False
        self._env_id = cfg.env_id
        # self._flat_obs = cfg.flat_obs
        self._view_size = cfg.view_size
        self._save_replay_gif = cfg.save_replay_gif
        self._replay_path_gif = cfg.replay_path_gif
        self._max_step = cfg.max_step
        self._save_replay_count = 0
        self._timestep = 0

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`np.ndarray`): Initial observation from the environment.
        """
        if not self._init_flag:
            if self._save_replay_gif:
                self._env = gym.make(self._env_id, render_mode="rgb_array", max_steps=self._max_step, agent_view_size=self._view_size)
            else:
                self._env = gym.make(self._env_id, max_steps=self._max_step)

            # if self._env_id in ['MiniGrid-AKTDT-13x13-v0' or 'MiniGrid-AKTDT-13x13-1-v0']:
            #     # customize the agent field of view size, note this must be an odd number
            #     # This also related to the observation space, see gym_minigrid.wrappers for more details
            #     self._env = ViewSizeWrapper(self._env, agent_view_size=5)
            # if self._env_id == 'MiniGrid-AKTDT-7x7-1-v0':
            #     self._env = ViewSizeWrapper(self._env, agent_view_size=3)

            # if self._view_size:
                # self._env = ViewSizeWrapper(self._env, agent_view_size=self._view_size)
            # if self._flat_obs:
            # self._env = FlatObsWrapper(self._env)
            self._env = ImgObsWrapper(self._env)
                # self._env = RGBImgPartialObsWrapper(self._env)
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._init_flag = True
        # if self._flat_obs:
        #     # Get the actual observation space from the wrapped environment
        #     # instead of using a hardcoded shape
        #     self._observation_space = self._env.observation_space
        # else:
        
        self._observation_space = self._env.observation_space
        # to be compatible with subprocess env manager
        if isinstance(self._observation_space, gym.spaces.Dict):
            self._observation_space['obs'].dtype = np.dtype('float32')
        else:
            self._observation_space.dtype = np.dtype('float32')

        self._action_space = self._env.action_space
        self._reward_space = gym.spaces.Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            obs, _ = self._env.reset(seed=self._seed)  # using the reset method of Gymnasium env
        elif hasattr(self, '_seed'):
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()

        # obs = to_ndarray(obs)
        
        # Flatten the 3 dimensional tensor into a single observation vector for MLP
        obs = obs.flatten()  
        obs = obs.astype(np.float32)
        
        # print(f"obs shape ndarray: {obs.shape}, obs type: {type(obs)}, dtype: {obs.dtype}, obs: {obs}")

        self._eval_episode_return = 0
        self._current_step = 0
        if self._save_replay_gif:
            self._frames = []

        action_mask = np.ones(self.action_space.n, 'int8')
        self._timestep = 0
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}

        return obs

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`np.ndarray`): The action to be performed in the environment. 
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if isinstance(action, np.ndarray) and action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        if self._save_replay_gif:
            img = self._env.render()
            enriched_img = self.add_info_to_image(img, action, self._timestep)
            self._frames.append(enriched_img)

        # using the step method of Gymnasium env, return is (observation, reward, terminated, truncated, info)
        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        rew = float(rew)
        self._eval_episode_return += rew
        self._current_step += 1
        if self._current_step >= self._max_step:
            done = True
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            info['current_step'] = self._current_step
            info['max_step'] = self._max_step
            if self._save_replay_gif:
                Path(self._replay_path_gif).mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                path = os.path.join(
                    self._replay_path_gif,
                    '{}_episode_{}_seed{}_{}.gif'.format(self._env_id, self._save_replay_count, self._seed, timestamp)
                )
                self.display_frames_as_gif(self._frames, path)
                print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
                self._save_replay_count += 1
        obs = to_ndarray(obs)
        rew = to_ndarray(rew)  # wrapped to be transferred to an array with shape (1,)

        action_mask = np.ones(self.action_space.n, 'int8')
        self._timestep += 1

        # Flatten the 3 dimensional tensor into a single observation vector for MLP
        obs = obs.flatten()
        obs = obs.astype(np.float32)

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}

        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        """
         Generate a random action using the action space's sample method. Returns a numpy array containing the action.
        """
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable the saving of replay videos. If no replay path is given, a default is used.
        """
        if replay_path is None:
            replay_path = './video'
        self._save_replay = True
        self._replay_path = replay_path
        self._save_replay_count = 0

    @staticmethod
    def add_info_to_image(img: np.ndarray, action: int, timestep: int) -> np.ndarray:
        """
        Add action text and timestep to the corner of the rendered image.
        
        Args:
            img: The original image as numpy array
            action: The action index
            timestep: The current timestep number
            
        Returns:
            Modified image with action text and timestep in the corner
        """
       
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Define action names for MiniGrid
        action_names = ['<', '>', '^', 'pickup', 'drop', 'toggle', 'done']
        action_name = action_names[action] if action < len(action_names) else f'action_{action}'
        action_text = f'Action: {action_name}'
        timestep_text = f'ts: {timestep}'
        
        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding boxes
        action_bbox = draw.textbbox((0, 0), action_text, font=font)
        action_width = action_bbox[2] - action_bbox[0]
        action_height = action_bbox[3] - action_bbox[1]
        
        timestep_bbox = draw.textbbox((0, 0), timestep_text, font=font)
        timestep_width = timestep_bbox[2] - timestep_bbox[0]
        timestep_height = timestep_bbox[3] - timestep_bbox[1]
        
        img_width = img_pil.width
        padding = 5
        
        # Position action text in top left corner
        action_x = padding
        action_y = padding
        
        # Position timestep text in top right corner
        timestep_x = img_width - timestep_width - padding
        timestep_y = padding
        
        # Add black background rectangle for action text
        draw.rectangle(
            [(action_x - padding, action_y - padding), (action_x + action_width + padding, action_y + action_height + padding)], 
            fill=(0, 0, 0)
        )
        
        # Add black background rectangle for timestep text
        draw.rectangle(
            [(timestep_x - padding, timestep_y - padding), (timestep_x + timestep_width + padding, timestep_y + timestep_height + padding)], 
            fill=(0, 0, 0)
        )
        
        # Add the action text in white (top left)
        draw.text((action_x, action_y), action_text, fill=(255, 255, 255), font=font)
        
        # Add the timestep text in white (top right)
        draw.text((timestep_x, timestep_y), timestep_text, fill=(255, 255, 255), font=font)
        
        # Convert back to numpy array
        return np.array(img_pil)

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero MiniGrid Env({})".format(self._cfg.env_id)