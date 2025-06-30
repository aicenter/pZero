from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from lzero.model.common import MZNetworkOutput, RepresentationNetworkMLP, PredictionNetworkMLP, MLP_V2
from lzero.model.utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('POMuZeroModelMLP')
class POMuZeroModelMLP(nn.Module):

    def __init__(
        self,
        observation_shape: int = 2,
        action_space_size: int = 6,
        latent_state_dim: int = 256,
        reward_head_hidden_channels: SequenceType = [32],
        value_head_hidden_channels: SequenceType = [32],
        policy_head_hidden_channels: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        discrete_action_encoding_type: str = 'one_hot',
        norm_type: Optional[str] = 'BN',
        res_connection_in_dynamics: bool = False,

        # New parameters for recursive representation
        use_recursive_representation: bool = False,
        recursive_repr_hidden_channels: int = 256,
        recursive_repr_layer_num: int = 2,
        res_connection_in_recursive_repr: bool = False,
        learned_initial_state: bool = False,
        
        *args,
        **kwargs
    ):
        """
        Overview:
            The definition of the network model of MuZero, which is a generalization version for 1D vector obs.
            The networks are mainly built on fully connected layers.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP network which predicts the next latent state, and reward given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, e.g. 4 for Lunarlander.
            - latent_state_dim (:obj:`int`): The dimension of latent state, such as 256.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            - discrete_action_encoding_type (:obj:`str`): The encoding type of discrete action, which can be 'one_hot' or 'not_one_hot'.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
            - use_recursive_representation (:obj:`bool`): Whether to use recursive representation network instead of standard representation, default set it to False.
            - recursive_repr_hidden_channels (:obj:`int`): The number of hidden channels in recursive representation network, default set it to 256.
            - recursive_repr_layer_num (:obj:`int`): The number of layers in recursive representation network, default set it to 2.
            - res_connection_in_recursive_repr (:obj:`bool`): Whether to use residual connection in recursive representation network, default set it to False.
            - learned_initial_state (:obj:`bool`): Whether to use learned initial state instead of zero initialization, default set it to False.
        """
        super(POMuZeroModelMLP, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.action_space_size = action_space_size
        self.continuous_action_space = False
        # The dim of action space. For discrete action space, it is 1.
        # For continuous action space, it is the dimension of continuous action.
        self.action_space_dim = action_space_size if self.continuous_action_space else 1
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.continuous_action_space:
            self.action_encoding_dim = action_space_size
        else:
            if self.discrete_action_encoding_type == 'one_hot':
                self.action_encoding_dim = action_space_size
            elif self.discrete_action_encoding_type == 'not_one_hot':
                self.action_encoding_dim = 1

        self.latent_state_dim = latent_state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics
        
        # New recursive representation attributes
        self.use_recursive_representation = use_recursive_representation
        self.recursive_repr_hidden_channels = recursive_repr_hidden_channels
        self.recursive_repr_layer_num = recursive_repr_layer_num
        self.res_connection_in_recursive_repr = res_connection_in_recursive_repr
        self.learned_initial_state = learned_initial_state

        # Initialize representation networks
        if self.use_recursive_representation:
            self.recursive_representation_network = RecursiveRepresentationNetwork(
                observation_shape=observation_shape,
                latent_state_dim=self.latent_state_dim,
                hidden_channels=self.recursive_repr_hidden_channels,
                layer_num=self.recursive_repr_layer_num,
                activation=activation,
                norm_type=norm_type,
                last_linear_layer_init_zero=False,
                res_connection=self.res_connection_in_recursive_repr,
            )
            
            # Initialize initial state
            if self.learned_initial_state:
                self.initial_state_network = MLP(
                    in_channels=observation_shape,
                    hidden_channels=self.latent_state_dim,
                    out_channels=self.latent_state_dim,
                    layer_num=2,
                    activation=activation,
                    norm_type=norm_type,
                    output_activation=False,
                    output_norm=False,
                    last_linear_layer_init_zero=False,
                )
            else:
                # Use learnable parameter for initial state - initialize with small random values
                self.initial_latent_state = nn.Parameter(torch.randn(self.latent_state_dim) * 0.01)
        else:
            # Keep original representation network for backward compatibility
            self.representation_network = RepresentationNetworkMLP(
                observation_shape=observation_shape, hidden_channels=self.latent_state_dim, norm_type=norm_type
            )

        self.dynamics_network = DynamicsNetwork(
            action_encoding_dim=self.action_encoding_dim,
            num_channels=self.latent_state_dim + self.action_encoding_dim,
            common_layer_num=2,
            reward_head_hidden_channels=reward_head_hidden_channels,
            output_support_size=self.reward_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type,
            res_connection_in_dynamics=self.res_connection_in_dynamics,
        )

        self.prediction_network = PredictionNetworkMLP(
            action_space_size=action_space_size,
            num_channels=latent_state_dim,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            output_support_size=self.value_support_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            norm_type=norm_type
        )
        

        if self.self_supervised_learning_loss:
            # self_supervised_learning_loss related network proposed in EfficientZero
            self.projection_input_dim = latent_state_dim

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of MuZero model, which is the first step of the MuZero model.
            For recursive representation: combines observation with initial latent state to get first latent state.
            For standard representation: uses representation network to obtain latent state from observation.
            Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        batch_size = obs.size(0)
        
        if self.use_recursive_representation:
            # Use recursive representation with initial latent state
            if self.learned_initial_state:
                initial_state = self.initial_state_network(obs)
            else:
                initial_state = self.initial_latent_state.unsqueeze(0).repeat(batch_size, 1)
            
            latent_state = self._recursive_representation(obs, initial_state)
        else:
            # Use standard representation network
            latent_state = self._representation(obs)
            
        policy_logits, value = self._prediction(latent_state)
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of MuZero model, which is the rollout step of the MuZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward`` by the given current ``latent_state`` and ``action``.
             We then use the prediction network to predict the ``value`` and ``policy_logits``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input obs.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        next_latent_state, reward = self._dynamics(latent_state, action)
        policy_logits, value = self._prediction(next_latent_state)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def observation_inference(self, obs: torch.Tensor, prev_latent_state: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Observation inference for recursive representation network.
            This is used when a new observation is received and we need to update the latent state
            based on both the observation and the previous latent state.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
            - prev_latent_state (:obj:`torch.Tensor`): The previous latent state.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): Set to zero for observation-driven transitions.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The new latent state after processing observation.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - prev_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        if not self.use_recursive_representation:
            raise ValueError("observation_inference can only be used with recursive representation enabled")
            
        batch_size = obs.size(0)
        latent_state = self._recursive_representation(obs, prev_latent_state)
        policy_logits, value = self._prediction(latent_state)
        
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def _representation(self, observation: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Overview:
             Use the representation network to encode the observations into latent state.
        Arguments:
             - obs (:obj:`torch.Tensor`): The 1D vector  observation data.
        Returns:
             - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
             - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
         """
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state

    def _recursive_representation(self, observation: torch.Tensor, prev_latent_state: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Use the recursive representation network to encode observations combined with previous latent state.
        Arguments:
            - observation (:obj:`torch.Tensor`): The 1D vector observation data.
            - prev_latent_state (:obj:`torch.Tensor`): The previous latent state.
        Returns:
            - latent_state (:obj:`torch.Tensor`): The new latent state after processing observation and previous state.
        Shapes:
            - observation (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - prev_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        if not self.use_recursive_representation:
            raise ValueError("_recursive_representation can only be used with recursive representation enabled")
        
        # Concatenate observation and previous latent state
        obs_state_encoding = torch.cat([observation, prev_latent_state], dim=1)
        
        # Pass through recursive representation network
        latent_state = self.recursive_representation_network(obs_state_encoding)
        
        if self.state_norm:
            latent_state = renormalize(latent_state)
            
        return latent_state

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the prediction network to predict the value and policy given the current latent state.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input observation.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        policy_logits, value = self.prediction_network(latent_state)
        return policy_logits, value

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            ``reward`` and ``next_reward_hidden_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        if self.discrete_action_encoding_type == 'one_hot':
            # Stack latent_state with the one hot encoded action
            if len(action.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action = action.unsqueeze(-1)

            # transform action to one-hot encoding.
            # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
            action_one_hot = torch.zeros(action.shape[0], self.action_space_size, device=action.device)
            # transform action to torch.int64
            action = action.long()
            action_one_hot.scatter_(1, action, 1)
            action_encoding = action_one_hot
        elif self.discrete_action_encoding_type == 'not_one_hot':
            action_encoding = action / self.action_space_size
            if len(action_encoding.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action_encoding = action_encoding.unsqueeze(-1)

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim]) or
        # (batch_size, latent_state[1] + action_space_size]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)

        if not self.state_norm:
            return next_latent_state, reward
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, reward

    def project(self, latent_state: torch.Tensor, with_grad=True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is proposed in EfficientZero.
            For more details, please refer to the paper ``Exploring Simple Siamese Representation Learning``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - with_grad (:obj:`bool`): Whether to calculate gradient for the projection result.
        Returns:
            - proj (:obj:`torch.Tensor`): The result embedding vector of projection operation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - proj (:obj:`torch.Tensor`): :math:`(B, projection_output_dim)`, where B is batch_size.

        Examples:
            >>> latent_state = torch.randn(256, 64)
            >>> output = self.project(latent_state)
            >>> output.shape # (256, 1024)
        """
        proj = self.projection(latent_state)

        if with_grad:
            # with grad, use prediction_head
            return self.prediction_head(proj)
        else:
            return proj.detach()

    def get_params_mean(self) -> float:
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):

    def __init__(
        self,
        action_encoding_dim: int = 2,
        num_channels: int = 64,
        common_layer_num: int = 2,
        reward_head_hidden_channels: SequenceType = [32],
        output_support_size: int = 601,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        res_connection_in_dynamics: bool = False,
    ):
        """
        Overview:
            The definition of dynamics network in MuZero algorithm, which is used to predict next latent state
            reward by the given current latent state and action.
            The networks are mainly built on fully connected layers.
        Arguments:
            - action_encoding_dim (:obj:`int`): The dimension of action encoding.
            - num_channels (:obj:`int`): The num of channels in latent states.
            - common_layer_num (:obj:`int`): The number of common layers in dynamics network.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection in dynamics network.
        """
        super().__init__()
        assert num_channels > action_encoding_dim, f'num_channels:{num_channels} <= action_encoding_dim:{action_encoding_dim}'

        self.num_channels = num_channels
        self.action_encoding_dim = action_encoding_dim
        self.latent_state_dim = self.num_channels - self.action_encoding_dim

        self.res_connection_in_dynamics = res_connection_in_dynamics
        if self.res_connection_in_dynamics:
            self.fc_dynamics_1 = MLP(
                in_channels=self.num_channels,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
            self.fc_dynamics_2 = MLP(
                in_channels=self.latent_state_dim,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )
        else:
            self.fc_dynamics = MLP(
                in_channels=self.num_channels,
                hidden_channels=self.latent_state_dim,
                layer_num=common_layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                # last_linear_layer_init_zero=False is important for convergence
                last_linear_layer_init_zero=False,
            )

        self.fc_reward_head = MLP_V2(
            in_channels=self.latent_state_dim,
            hidden_channels=reward_head_hidden_channels,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the dynamics network. Predict the next latent state given current latent state and action.
        Arguments:
            - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, latent_state_dim).
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        """
        if self.res_connection_in_dynamics:
            # take the state encoding (e.g. latent_state),
            # state_action_encoding[:, -self.action_encoding_dim:] is action encoding
            latent_state = state_action_encoding[:, :-self.action_encoding_dim]
            x = self.fc_dynamics_1(state_action_encoding)
            # the residual link: add the latent_state to the state_action encoding
            next_latent_state = x + latent_state
            next_latent_state_encoding = self.fc_dynamics_2(next_latent_state)
        else:
            next_latent_state = self.fc_dynamics(state_action_encoding)
            next_latent_state_encoding = next_latent_state

        reward = self.fc_reward_head(next_latent_state_encoding)

        return next_latent_state, reward

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> float:
        return get_reward_mean(self)


class RecursiveRepresentationNetwork(nn.Module):

    def __init__(
        self,
        observation_shape: int,
        latent_state_dim: int,
        hidden_channels: int = 256,
        layer_num: int = 2,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        last_linear_layer_init_zero: bool = True,
        res_connection: bool = False,
    ):
        """
        Overview:
            The definition of recursive representation network, which takes observation and previous latent state
            to predict the new latent state. Similar to DynamicsNetwork but for observation-driven transitions.
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space.
            - latent_state_dim (:obj:`int`): The dimension of latent state.
            - hidden_channels (:obj:`int`): The number of hidden channels in MLP.
            - layer_num (:obj:`int`): The number of layers in MLP.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network.
            - norm_type (:obj:`str`): The type of normalization in networks.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer.
            - res_connection (:obj:`bool`): Whether to use residual connection.
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.latent_state_dim = latent_state_dim
        self.input_dim = observation_shape + latent_state_dim  # concatenate obs and prev_latent_state
        self.res_connection = res_connection
        
        if self.res_connection:
            self.fc_1 = MLP(
                in_channels=self.input_dim,
                hidden_channels=hidden_channels,
                layer_num=layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                last_linear_layer_init_zero=False,
            )
            self.fc_2 = MLP(
                in_channels=self.latent_state_dim,
                hidden_channels=hidden_channels,
                layer_num=layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                last_linear_layer_init_zero=False,
            )
        else:
            self.fc_recursive_repr = MLP(
                in_channels=self.input_dim,
                hidden_channels=hidden_channels,
                layer_num=layer_num,
                out_channels=self.latent_state_dim,
                activation=activation,
                norm_type=norm_type,
                output_activation=True,
                output_norm=True,
                last_linear_layer_init_zero=last_linear_layer_init_zero,
            )
        
        # Initialize weights properly to prevent exploding gradients
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent exploding gradients."""
        def init_recursive_module(module):
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with smaller scale
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        if self.res_connection:
            self.fc_1.apply(init_recursive_module)
            self.fc_2.apply(init_recursive_module)
        else:
            self.fc_recursive_repr.apply(init_recursive_module)

    def forward(self, obs_state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward computation of the recursive representation network.
            Predict the new latent state given observation and previous latent state.
        Arguments:
            - obs_state_encoding (:obj:`torch.Tensor`): The concatenation of observation and previous latent state,
                with shape (batch_size, observation_shape + latent_state_dim).
        Returns:
            - new_latent_state (:obj:`torch.Tensor`): The new latent state, with shape (batch_size, latent_state_dim).
        """
        if self.res_connection:
            # Extract previous latent state for residual connection
            prev_latent_state = obs_state_encoding[:, -self.latent_state_dim:]
            x = self.fc_1(obs_state_encoding)
            # Residual connection: add previous latent state
            new_latent_state = x + prev_latent_state
            new_latent_state = self.fc_2(new_latent_state)
        else:
            new_latent_state = self.fc_recursive_repr(obs_state_encoding)
        
        # Reduced scaling to prevent exploding gradients
        # Use a gentler activation with balanced scaling factor
        new_latent_state = torch.tanh(new_latent_state) * 1.0  # Increased from 0.5 to 1.0 for better learning
        
        return new_latent_state
