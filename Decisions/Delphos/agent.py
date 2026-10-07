from curses.ascii import alt
import datetime
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

import delphos

# Example usage:
# from agent import DQNLearner
# agent = DQNLearner(
#     path_rewards='results/',
#     path_choice_dataset='data.csv',
#     path_to_save='experiments/',
#     state_space_params={'num_vars': 7, 'transformations': ['linear', 'log'], 'taste': ['generic'], 'covariates': []},
#     num_episodes=10000,
#     attributes={i: set(range(1, 6)) for i in range(1, 4)},
#     covariates={}
# )
# agent.train()


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


########################################################################################
# Core classes: DQNetwork, ReplayBuffer, StateManager
########################################################################################

class DQNetwork(nn.Module):
    """
    Deep Q-Network with a variable number of hidden layers.

    Args:
        input_size (int): Size of the input tensor.
        output_size (int): Size of the output tensor.
        hidden_layers (List[int], optional): List of hidden layer sizes. Defaults to [128, 64].
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int] = [128, 64]) -> None:
        super().__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling past experiences.

    Args:
        max_size (int): Maximum number of experiences to store.
    """
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, transition: Tuple) -> None:
        """
        Add a new experience to the buffer.

        Args:
            transition (Tuple): Experience tuple to add.
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Any:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Any: Batch of sampled experiences, or None if not enough data.
        """
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class StateManager:    
    """
    Manages the encoding and decoding of state representations for model specifications.

    Args:
        state_space_params (Dict[str, Any]): Parameters defining the state space.
    """
    def __init__(self, state_space_params: Dict[str, Any]) -> None:
        default_trans = ['linear', 'log', 'box-cox']
        default_taste = ['generic', 'specific']
        self.state_space_params = {
            'num_vars': state_space_params.get('num_vars', 1),
            'transformations': state_space_params.get('transformations', default_trans),
            'taste': state_space_params.get('taste', default_taste),
            'covariates': state_space_params.get('covariates', [])
        }
        self.transformation_codes = {0: 'none', 1: 'linear', 2: 'log', 3: 'box-cox'}
        self.inverse_mapping = {v: k for k, v in self.transformation_codes.items()}

    def get_state_length(self) -> int:
        """
        Compute the length of the state representation vector used for encoding model specifications.

        Returns:
            int: Total number of dimensions in the one-hot encoded state vector.
        """
        asc = 1
        att = self.state_space_params['num_vars'] - asc
        att_ = 4  
        taste_ = 2  
        cov_ = len(self.state_space_params['covariates']) + 1
        if 'specific' in self.state_space_params['taste']:
            return (asc * cov_) + (att * att_ * taste_ * cov_)
        else:
            return (asc * cov_) + (att * att_)

    def encode_state_to_vector(self, state: List[Tuple]) -> torch.FloatTensor:
        """
        Encode a model specification into a one-hot vector representation.

        This encoding is used as input to the neural network model. The encoding format depends on whether
        specific taste parameters and/or covariates are present in the model.
       
        Args:
            state (List[Tuple]): A list of tuples representing the model specification.

            Examples:
                - No specific/covariates: (var, trans)
                - With specific: (var, trans, spec)
                - With specific and covariates: (var, trans, spec, cov)

        Returns:
            torch.FloatTensor: A one-hot encoded state vector representing the specification.
        """
        state_vector = np.zeros(self.get_state_length())
        transformations = self.state_space_params['transformations']
        covariates = self.state_space_params['covariates']
        num_transformations = 4
        num_specific = 2
        num_covariates = len(covariates) + 1
        if not 'specific' in self.state_space_params['taste']:
            asc_offset = 1
            for var, trans in state:
                if var == 0 and trans == 'linear':
                    state_vector[0] = 1
                else:
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    attr_idx = (var - 1) * num_transformations
                    state_vector[asc_offset + attr_idx + trans_idx] = 1
        elif 'specific' in self.state_space_params['taste'] and not covariates:
            for var, trans, spec in state:
                if var == 0 and trans == 'linear' and spec == 0:
                    state_vector[0] = 1
                else:
                    attr_idx = (var - 1) * (num_transformations * num_specific)
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    spec_idx = spec
                    total = attr_idx + (1 - spec_idx) * trans_idx + spec_idx * (num_transformations + trans_idx)
                    state_vector[total] = 1
        else:
            for var, trans, spec, cov in state:
                if var == 0 and trans == 'linear' and spec == 0:
                    state_vector[cov] = 1
                else:
                    asc_offset = 2  # ASC_000, ASC_100, ASC_101, ASC_102 occupy the first positions
                    attr_idx = (var - 1) * (num_transformations * num_specific * num_covariates)
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    spec_idx = spec
                    cov_idx = cov
                    total = asc_offset + attr_idx + (1 - spec_idx) * trans_idx + spec_idx * (num_transformations + trans_idx) + cov_idx * (num_transformations * num_specific)
                    state_vector[total] = 1
        return torch.FloatTensor(state_vector)

    def encode_state_to_string(self, state: List[Tuple]) -> str:
        """
        Encode a state tuple into a compact string representation used by Delphos.
        The string is formed by concatenating numeric codes for transformation, taste, and covariate indicators for each variable.

        Args:
            state (List[Tuple]): List of tuples representing the specification (e.g., [(0, 'linear', 0, 0), ...]).

        Returns:
            str: A string representation of the state (e.g., '100_000_311').
        """
        representation = ['000'] * self.state_space_params['num_vars']
        covariates = self.state_space_params['covariates']
        use_specific = 'specific' in self.state_space_params['taste']
        for entry in state:
            #if use_specific and covariates:
            #    var, trans, spec, cov = entry
            #elif use_specific:
            #    var, trans, spec = entry
            #    cov = 0
            #else:
            #    var, trans = entry
            #    spec, cov = 0, 0
            
            if use_specific and covariates:
                var, trans, spec, cov = entry
            
            elif not use_specific and covariates:
                var, trans, cov = entry
                spec = 0
            
            elif use_specific and not covariates:
                var, trans, spec = entry
                cov = 0

            elif not use_specific and not covariates:
                var, trans = entry
                spec, cov = 0, 0

            elif use_specific:
                var, trans, spec = entry
                cov = 0                
            else:
                var, trans = entry
                spec, cov = 0, 0

            trans_code = self.inverse_mapping.get(trans, 0)
            representation[var] = f"{trans_code}{spec}{cov}".zfill(3)
        return '_'.join(representation)

    def decode_string_to_state(self, representation: str) -> List[Tuple]:
        """
        Decode a Delphos-style model string representation into a structured state format (list of tuples).

        Args:
            representation (str): Delphos string representation (e.g., '100_000_311').

        Returns:
            List[Tuple]: List of tuples encoding the specification as used internally by the agent (e.g., [(0, 'linear', 0, 0), ...]).
        """
        covariates = self.state_space_params['covariates']
        use_specific = 'specific' in self.state_space_params['taste']
        rep_list = representation.split('_')
        state = []
        for idx, code in enumerate(rep_list):
            if code == '000':
                continue
            trans_code = int(code[0])
            spec = int(code[1]) if use_specific else 0
            cov = int(code[2]) if covariates else 0
            trans = self.transformation_codes.get(trans_code, 'none')
            if use_specific and covariates:
                state.append((idx, trans, spec, cov))
            elif use_specific:
                state.append((idx, trans, spec))
            else:
                state.append((idx, trans))
        return state
    
    def decode_string_to_specification(self, representation: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Decode a model string into separate component vectors for transformation, taste, and covariates.

        Args:
            representation (str): Delphos string representation (e.g., '100_000_311').

        Returns:
            Tuple[List[int], List[int], List[int]]: Three lists containing transformation, specific taste, and covariate respectively (e.g., [1, 0, 3], [0, 0, 1], [0, 0, 1]).
        """
        state_0, specific_0, covariates_0 = [], [], []
        for t in representation.split('_'):
            if len(t) >= 3:
                state_0.append(int(t[0]))
                specific_0.append(int(t[1]))
                covariates_0.append(int(t[2]))
            else:
                raise ValueError(f"Invalid representation: {representation}")
        return state_0, specific_0, covariates_0
    
    def mask_invalid_actions(self, state: List[Tuple], action_space: List[Tuple]) -> List[Tuple]:
        """
        Mask invalid actions based on the current state and action space.

        Args:
            state (List[Tuple]): Current state as a list of tuples.
            action_space (List[Tuple]): List of possible actions.

        Returns:
            List[Tuple]: List of valid actions based on the current state.
        """
        transformations = self.state_space_params['transformations']
        covariates = self.state_space_params['covariates']
        taste = self.state_space_params['taste']
        use_specific = 'specific' in taste
        asc_added = any(var == 0 for var, *rest in state)
        valid_actions = []
        add_current_attributes = set()
        change_current_attributes = set()
        if not use_specific and not covariates:
            add_current_attributes = {var for var, trans in state}
            change_current_attributes = {(var, trans) for var, trans in state}
        elif use_specific and not covariates:
            add_current_attributes = {var for var, trans, spec in state}
            change_current_attributes = {(var, trans, spec) for var, trans, spec in state}
        elif use_specific and covariates:
            add_current_attributes = {var for var, trans, spec, cov in state}
            change_current_attributes = {(var, trans, spec, cov) for var, trans, spec, cov in state}
        seen_asc_action = False
        for action in action_space:
            act_type = action[0]
            if act_type == 'terminate':
                valid_actions.append(action)
                continue
            var = action[1]
            trans = action[2]
            if var == 0:
                if (use_specific and len(action) < 4) or (len(covariates)>0 and len(action) <5) :
                    continue

                if asc_added or seen_asc_action:
                    if act_type == 'add':
                            continue        
                    else:
                        if (var, trans, *action[3:]) in change_current_attributes:
                            continue
                        else:
                            valid_actions.append(action)
                        
                        seen_asc_action = True
            if not use_specific and not covariates:
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans) not in change_current_attributes:
                    valid_actions.append(action)
            elif use_specific and not covariates:
                spec = action[3]
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans, spec) not in change_current_attributes:
                    valid_actions.append(action)
            elif use_specific and covariates:
                spec = action[3]
                cov = action[4]
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans, spec, cov) not in change_current_attributes:
                    valid_actions.append(action)
        valid_actions = list(dict.fromkeys(valid_actions))  
        return valid_actions

########################################################################################
# RL Agent: DQNLearner
########################################################################################

class DQNLearner:
    """
    DQN agent for model specification with experience replay, candidate tracking, and early stopping.

    Args:
        path_rewards (str): Path to save reward logs.
        path_choice_dataset (str): Path to the choice dataset.
        path_to_save (str): Directory to save outputs.
        state_space_params (Dict[str, Any]): Parameters defining the state space.
        num_episodes (int): Total number of training episodes.
        attributes (Dict[int, set]): Attributes per alternative.
        covariates (Dict[str, List[int]]): Covariate categories.
        reward_metrics (List[str], optional): List of reward metrics to optimize (default: ['adjRho2_0']).
        
    Other Args:
        agent_index (int, optional): Agent identifier.
        discount_factor (float, optional): Discount factor gamma.
        learning_rate (float, optional): Learning rate.
        buffer_size (int, optional): Replay buffer size.
        batch_size (int, optional): Batch size.
        target_update_freq (int, optional): Target network update frequency.
        batch_log_size (int, optional): Log print frequency.
        patience (int, optional): Early stopping patience.
        early_stop_window (int, optional): Rolling mean window for early stopping.
        early_stop_tolerance (float, optional): Early stopping improvement threshold.
        min_percentage (float, optional): Minimum percentage of episodes before early stopping.
        reward_weights (Dict[str, float], optional): Weights for each reward metric.
        reward_distribution (str, optional): Reward distribution strategy.

    
    """
    def __init__(self, path_rewards: str, path_choice_dataset: str, path_to_save: str,
                 state_space_params: Dict[str, Any], num_episodes: int,
                 attributes: Dict[int, set], covariates: Dict[str, List[int]], agent_index: int = 1,
                 discount_factor: float = 0.9, learning_rate: float = 0.001, buffer_size: int = 10000,
                 batch_size: int = 64, target_update_freq: int = 10, batch_log_size: int = 64,
                 patience: int = 100, early_stop_window: int = 500, early_stop_tolerance: float = 0.001,
                 min_percentage: float = 0.5, reward_weights=None, reward_distribution: str = 'exponential', hidden_layers: List[int] = [128, 64], behavioural_expectation = {} ) -> None:
        
        super(DQNLearner, self).__init__()

        logger.info('Initializing DQNLearner')
        logger.info('Number of CPUs: %d', mp.cpu_count())        
        self.state_space_params = state_space_params
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.batch_log_size = batch_log_size
        self.target_update_freq = target_update_freq
        self.epsilon = 1.0
        self.epsilon_decay = 1.0 / (num_episodes * min_percentage)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_manager = StateManager(state_space_params) 

        logger.info('State space parameters: %s', state_space_params)
        logger.info('Number of episodes: %d', self.num_episodes)
        logger.info('Batch size: %d', self.batch_size)
        logger.info('Buffer size: %d', buffer_size)
        logger.info('Target update frequency: %d', self.target_update_freq)
        logger.info('Replay buffer size: %d', buffer_size)

        self.path_rewards = path_rewards  
        self.path_choice_dataset = path_choice_dataset
        self.path_to_save = path_to_save
        self.attributes = attributes
        self.covariates = covariates
        self.agent_index = agent_index
        
        # Reward metrics and weights 
        self.reward_weights = reward_weights or {'adjRho2_0': 1.0}
        default_metric_directions = {'AIC': 'minimize','BIC': 'minimize','adjRho2_0': 'maximize','rho2_0': 'maximize','rho2_C': 'maximize','adjRho2_C': 'maximize','LLout': 'maximize', 'numParams': 'minimize'}
        self.metric_directions = {metric: default_metric_directions.get(metric, 'maximize') for metric in self.reward_weights}
        self.metric = list(self.reward_weights.keys())[0]
        self.reward_distribution = reward_distribution
        self.behavioural_expectation = behavioural_expectation  
        
        

        ### Reward database ###
        job_results_file = os.path.join(self.path_rewards, "outputs", "rewards.csv")
        if os.path.exists(job_results_file):
            self.rewards = pd.read_csv(job_results_file)
        else:
            self.rewards = pd.DataFrame(columns=["specification"])

        ### Action space ###
        self.action_space, self.action_to_index = self.define_action_space()

        self.action_log = []        
        self.buffer_log = []
        self.training_log = []
        self.date = pd.Timestamp("today").strftime("%Y_%m_%d_%H_%M") 

        # Initialize the policy and target networks
        input_size = self.state_manager.get_state_length()
        output_size = len(self.action_space)
        self.policy_net = DQNetwork(input_size, output_size, hidden_layers=hidden_layers)
        self.target_net = DQNetwork(input_size, output_size, hidden_layers=hidden_layers)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()       

        if not os.path.exists(self.path_to_save):
            logger.info('Creating folder %s', self.path_to_save)
            os.makedirs(self.path_to_save, exist_ok=True)
        else:
            logger.info('Folder %s already exists', self.path_to_save)

        folder_count = len([name for name in os.listdir(self.path_to_save) if os.path.isdir(os.path.join(self.path_to_save, name))])
        logger.info('Number of files in the folder: %d', folder_count)
        folder_count +=  1

        self.subfolder = f'{self.path_to_save}/iteration_{folder_count}'
        if not os.path.exists(self.subfolder):
            logger.info('Creating subfolder %s', self.subfolder)
            os.makedirs(self.subfolder, exist_ok=True)

        logger.info('Number of episodes: %d | Number of batches: %d', self.num_episodes, self.num_episodes // self.batch_log_size)

        # Candidate tracking for improved learning
        self.best_candidates = {metric: {'value': -np.inf if self.metric_directions[metric] == 'maximize' else np.inf, 'episode': -1,'representation': None}  for metric in self.reward_weights}
        obs = len(pd.read_csv(os.path.join(self.path_rewards, self.path_choice_dataset))['ID'])*0.8
        alt = max(self.attributes.keys())
        self.LL0 = np.log(1/alt) * obs
        self.LL0 = -4768.431 

        logger.info("LL0: %f, AIC_0: %f, and BIC_0: %f", self.LL0, -2 * self.LL0, -2 * self.LL0)

        self.current_best_history = []        
        self.patience = patience
        self.early_stop_window = early_stop_window
        self.early_stop_tolerance = early_stop_tolerance 
        self.min_percentage = min_percentage
        self.min_episodes_before_stop = self.num_episodes*self.min_percentage
        self.no_improvement_count = 0     
        self.model_set = set()

        # Add file handler to save logs.
        log_file = os.path.join(self.subfolder, "agent_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def define_action_space(self) -> Tuple[List[Tuple], Dict[Tuple, int]]:
        """
        Define the action space based on the modeling space parameters.

        Returns:
            Tuple[List[Tuple], Dict[Tuple, int]]: (actions, action_to_index)
        """
        actions = [('terminate',)]        
        
        for var in range(self.state_space_params['num_vars']):
            if var == 0:  

                if 'specific' in self.state_space_params['taste'] and self.state_space_params['covariates'] != []:              
                    for cov in range(len(self.state_space_params['covariates'])+1):
                        actions.append(('add', var, 'linear', 0 , cov))
                        actions.append(('change', var, 'linear', 0, cov))

                elif 'specific' in self.state_space_params['taste'] and self.state_space_params['covariates'] == []:
                    for specific_flag in [0, 1]:  
                        actions.append(('add', var, 'linear', specific_flag))
                        actions.append(('change', var, 'linear', specific_flag))
            else:
                for transformation in self.state_space_params['transformations']:

                    if self.state_space_params['covariates']:  # When covariates are provided

                        if 'specific' in self.state_space_params['taste']:
                            # If "specific" taste is allowed, iterate over the binary flag (0 and 1)
                            for specific_flag in [0, 1]:  
                                for cov in range(len(self.state_space_params['covariates'])+1):
                                    actions.append(('add', var, transformation, specific_flag, cov))
                                    actions.append(('change', var, transformation, specific_flag, cov))
                        else:
                            # If "specific" taste is not allowed, always use a default flag (0)
                            for cov in range(len(self.state_space_params['covariates'])+1):
                                actions.append(('add', var, transformation, 0, cov))
                                actions.append(('change', var, transformation, 0, cov))

                    else:
                        # When no covariates are provided, omit the covariate element from the action tuple
                        if 'specific' in self.state_space_params['taste']:

                            for specific_flag in [0, 1]:  
                                actions.append(('add', var, transformation, specific_flag))
                                actions.append(('change', var, transformation, specific_flag))
                        else:
                            actions.append(('add', var, transformation))
                            actions.append(('change', var, transformation))
        
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        return actions, action_to_index   

    def select_action_index(self, state: List[Tuple]) -> int:       
        """
        Select an action using the epsilon-greedy policy.

        Args:
            state (List[Tuple]): The current state.

        Returns:
            int: The index of the chosen action.
        """
        valid_actions = self.state_manager.mask_invalid_actions(state, self.action_space)
        valid_action_indices = [self.action_to_index[action] for action in valid_actions] 
        
        if np.random.rand() < self.epsilon:
            return random.choice(valid_action_indices)
        else:
            with torch.no_grad():
                state_vector = self.state_manager.encode_state_to_vector(state)  # Convert state to vector form
                q_values = self.policy_net(state_vector)  # Get Q-values for all actions

                q_values_masked = torch.full((len(self.action_space),), float('-inf'))
                q_values_masked[valid_action_indices] = q_values[valid_action_indices]
                index = torch.argmax(q_values_masked).item()  # Choose the action with highest Q-value among valid ones
                return index
    
    def apply_action(self, state: List[Tuple], action_index: int) -> Tuple[List[Tuple], bool]:
        """
        Apply the chosen action to the current state.

        Args:
            state (List[Tuple]): The current state.
            action_index (int): The index of the action to apply.

        Returns:
            Tuple[List[Tuple], bool]: (next_state, done)
        """
        transformations = self.state_space_params['transformations']
        covariates = self.state_space_params['covariates']
        taste = self.state_space_params['taste']
        use_specific = 'specific' in taste

        state = list(state)
        action = self.action_space[action_index]
        
        if action[0] == 'terminate':
            return state, True                

        if use_specific and covariates == []:
            action_type, var, trans, spec = action        
            if action_type == 'add':
                state.append((var, trans, spec)) 
            elif action_type == 'change':
                state = [(v, t, s) if v != var else (v, trans, spec) for v, t, s in state]

        if use_specific and covariates != []:
            action_type, var, trans, spec, cov = action        
            if action_type == 'add':
                state.append((var, trans, spec, cov)) 
            elif action_type == 'change':
                state = [(v, t, s, c) if v != var else (v, trans, spec, cov) for v, t, s, c in state]
        
        if not use_specific and covariates == []:
            action_type, var, trans = action    
            if action_type == 'add':
                state.append((var, trans))
            elif action_type == 'change':
                state = [(v, t) if v != var else (v, trans) for v, t in state]
                            
        return state, False
    
    def generate_episode(self, episode_count: int) -> None:
        """
        Generate an episode of sequential decisions.

        Args:
            episode_count (int): The current episode number.

        Returns:
            Tuple[List[Tuple], List[Tuple], bool]: (episode_steps, final_state, done)
        """

        state, done, episode_steps = [], False, []

        #n_var = self.state_space_params['num_vars']
        #for i in range(n_var):
        #    state.append( (i, 'linear', 0, 0) )
        
        while not done:
            action_index = self.select_action_index(state)
            next_state, done = self.apply_action(state, action_index)
            episode_steps.append((state, action_index, next_state))
            self.action_log.append({'episode': episode_count, 'state': state, 'action': self.action_space[action_index], 'next_state': next_state})
            state = next_state
        
        return episode_steps, state, done

    def update_candidate_tracker(self, episode_count: int, modelling_outcomes: Dict[str, float], state: List[Tuple]) -> None:
        """
        Update best candidate models for each tracked metric based on modelling outcomes.

        Args:
            episode_count (int): Current episode number.
            modelling_outcomes (Dict[str, float]): Modelling outcomes from the estimation.
            state (List[Tuple]): Current state as a list of tuples.

        Notes:
            - The function checks if the modelling outcomes are valid (not NaN or infinite).
            - It updates the best candidates for each metric based on the specified direction (maximize/minimize).
            - The best candidates are stored in self.best_candidates.
            - A snapshot of the best candidates is saved in self.current_best_history.
        """
        best_repr = self.state_manager.encode_state_to_string(state)

        for metric in self.reward_weights:
            if metric in modelling_outcomes:
                value = modelling_outcomes[metric]
                if not isinstance(value, (int, float)) or pd.isna(value) or not np.isfinite(value):
                    continue
                current_best = self.best_candidates[metric]
                direction = self.metric_directions.get(metric, 'maximize')

                improved = (value > current_best['value'] if direction == 'maximize' else value < current_best['value'])

                if improved:
                    self.best_candidates[metric] = { 'value': value,'episode': episode_count,'representation': best_repr}
                    logger.info(f"New best candidate for {metric} at episode {episode_count}: {value:.4f} ({best_repr})")

        # Save a snapshot of the best candidates at this episode
        snapshot = {'episode': episode_count}
        for metric, info in self.best_candidates.items():
            snapshot[f"{metric}_value"] = info['value']
            snapshot[f"{metric}_repr"] = info['representation']
        self.current_best_history.append(snapshot)

    def check_early_stopping(self, episode_rewards: List[float]) -> bool:
        """
        Check early stopping condition based on the relative improvement of the rolling mean of episode rewards.

        Args:
            episode_rewards (List[float]): List of episode rewards.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if len(episode_rewards) < 2 * self.early_stop_window:
            return False
        current_window = episode_rewards[-self.early_stop_window:]
        previous_window = episode_rewards[-2*self.early_stop_window:-self.early_stop_window]
        current_mean = np.mean(current_window)
        previous_mean = np.mean(previous_window)
        # Relative improvement, add a small epsilon to avoid division by zero.
        relative_improvement = (current_mean - previous_mean) / (abs(previous_mean) + 1e-8)
        if relative_improvement < self.early_stop_tolerance:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        if self.no_improvement_count >= self.patience:
            return True
        return False

    def normalize_reward_metric(self, metric: str, value: float) -> float:
        """
        Normalize a modelling metric based on min/max scaling or predefined bounds.

        Args:
            metric (str): Name of the modelling outcome (e.g., 'LLout', 'AIC').
            value (float): Raw value from the estimation.

        Returns:
            float: Normalized modelling outcome in [0, 1].

        Notes:
            - LLout is normalized using LL0 as the lower bound and max observed as upper bound.
            - AIC/BIC are normalized assuming -2*LL0 is the theoretical maximum.
            - rho metrics are not normalized unless negative or non-finite.
        """
        values = [log[metric] for log in self.training_log if metric in log and pd.notna(log[metric])]
        max_val = max(values) if values else 0.0
        min_val = min(values) if values else 0.0
        
        if not values or self.LL0 is None:
            return 0.0

        norm = 0.0
        if metric == 'LLout':
            # Theoretical lower bound is self.LL0
            if value <= self.LL0 or max_val - self.LL0 == 0:
                return 0.0
            norm = (value - self.LL0) / (max_val - self.LL0)
        elif metric in ['AIC', 'BIC']:
            # Theoretical maximun for AIC/BIC is -2*self.LL0
            max_value = -2 * self.LL0
            if value >= max_value or max_val == min_val:
                return 0.0
            norm = (max_val - value) / (max_val - min_val)
        elif metric in ['rho2_0','adjRho2_0','rho2_C','adjRho2_C']:
            if value <= 0:
                return 0.0
            else:
                norm = value

        if np.isfinite(norm) and not pd.isna(norm):
            return float(norm)
        else:
            return 0.0

    def reward_function(self, modelling_outcome: pd.DataFrame) -> float:
        """
        Computes the final reward as a weighted sum of normalized modelling outcomes.

        Args:
            modelling_outcome (pd.DataFrame): Output from model estimation containing metrics.

        Returns:
            float: Final reward after applying weights and normalization to each metric.

        Notes:
            - If estimation was unsuccessful, returns 0.
            - Normalization follows min/max strategy adapted for each metric (LL, AIC/BIC, rho²).
        """
        try:
            if modelling_outcome.empty or modelling_outcome['successfulEstimation'].values[0] is False:
                return 0.0 
            
            if self.behavioural_expectation is not None: 
                for attr_index, expected_sign in self.behavioural_expectation.items():     
                    alternatives = self.attributes.keys()

                    generic_prefix = f"b_{attr_index}_generic"
                    specific_prefixes = [f"b_{alt}_{attr_index}_" for alt in self.attributes.keys()]
                    matching_cols = [col for col in modelling_outcome.columns if col.startswith(generic_prefix) or any(col.startswith(prefix) for prefix in specific_prefixes)]                   
                    for col in matching_cols:
                        try: 
                            val = float(modelling_outcome[col].values[0])
                            if expected_sign == '-' and val > 0:
                                return 0.0
                            if expected_sign == '+' and val < 0:
                                return 0.0
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid value for {col}")
                            return 0.0
                    
            reward = 0.0
            for metric, weight in self.reward_weights.items():
                if metric in modelling_outcome.columns:
                    raw_value = modelling_outcome[metric].values[0]
                    if not pd.isna(raw_value):
                        normalized_value = self.normalize_reward_metric(metric, raw_value)
                    else:
                        normalized_value = 0.0

                    reward += weight * normalized_value

            return reward
            
        except Exception as e:
            return 0.0 


    def delphos_interaction(self, state: List[Tuple], to_specification=False) -> Union[float, Tuple[str, float]]:
        """
        Interact with Delphos to obtain the reward signal for the given model specification.

        Args:
            state (List[Tuple]): The current state as a list of tuples.
                - Non-specific (no covariates): (var, transformation)
                - Specific (no covariates): (var, transformation, specific_flag)
                - Specific (with covariates): (var, transformation, specific_flag, cov)
            to_specification (bool, optional): Whether to return the model specification string along with the reward.

        Returns:
            Union[float, Tuple[str, float]]: The reward signal, or (specification, reward).

        Notes:
            - The function first checks if the model specification is already in the rewards database.
            - If not, it calls Delphos to estimate the model and obtain the reward signal.
            - The function also handles the case where LL0 is not provided in the modelling outcomes.
            - The function can return the model specification string for logging purposes.
        """
     
        representation = self.state_manager.encode_state_to_string(state)
        tuples = representation.split('_')

        state_0, specific_0, covariates_0 = [], [], []

        for tuple in tuples:
            if len(tuple) >= 3:
                state_0.append(int(tuple[0]))
                specific_0.append(int(tuple[1]))
                covariates_0.append(int(tuple[2]))
            else:
                raise ValueError(f'Invalid representation: {representation}')
        
        specification = [f"{state}{specific}{cov}" for state, specific, cov in zip(state_0, specific_0, covariates_0)]
        name = "_".join(specification)

        if name in self.rewards['specification'].values:
            modelling_outcomes = self.rewards[self.rewards['specification'] == name]
            
        else:
            modelling_outcomes = delphos.get_mnl_outcomes(state_0, specific_0, covariates_0, self.attributes, self.covariates, self.agent_index, self.path_rewards, self.path_choice_dataset,  r=True, info = False, episode= self.episode_count)
                
        final_reward = self.reward_function(modelling_outcomes)
        print(f"\t\tEpisode {self.episode_count}: Spec: {representation} | Reward: {final_reward:.4f} | LLout: {modelling_outcomes['LLout'].values[0] if 'LLout' in modelling_outcomes else 'N/A'}")
        if modelling_outcomes is not None:
            modelling_outcomes = modelling_outcomes[modelling_outcomes.columns.intersection(self.reward_weights.keys())]
        else:
            modelling_outcomes = pd.DataFrame()  
        
        # every 1000 episodes, update the reward file
        if self.episode_count % 1000 == 0:
            self.rewards = pd.read_csv(os.path.join(self.path_rewards, "outputs", "rewards.csv"))
            
        if to_specification:
            return representation, final_reward, modelling_outcomes
        return final_reward, modelling_outcomes
    
    def perform_experience_replay(self) -> None:
        """
        Sample a batch of transitions from the replay buffer and update the Q-network.

        Notes:
            - The function samples a batch of transitions from the replay buffer.
            - It computes the Q-values and target Q-values for the sampled transitions.
            - The loss is computed using mean squared error (MSE) between Q-values and target Q-values.
            - The optimizer updates the policy network based on the computed loss.
        """

        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return
        

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch)
        done_batch = torch.BoolTensor(done_batch)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (1 - done_batch.float()) * self.discount_factor * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self) -> None:
        """
        Train the DQN agent using the specified number of episodes and hyperparameters.
        
        Notes:
            - The training loop iterates over the specified number of episodes.
            - For each episode, it generates a sequence of decisions and computes rewards.
            - The replay buffer is updated with the transitions from the episode.
            - The Q-network is updated using experience replay.
            - The target network is updated periodically.
            - Early stopping conditions are checked based on the rolling mean of episode rewards.
            - The learning curve is plotted and saved.
            - The candidate history is saved and plotted.
            - The training logs are saved to the specified directory.
        """
        logger.info("Training agent %d", self.agent_index)
        start_time = time.time()
        logger.info("Training started at %s", time.ctime(start_time))
        episode_count = 0
        episode_rewards = []
        self.current_best_history = []
        early_stop_triggered = False

        for episode_count in range(self.num_episodes):
            # Generate an episode of sequential decisions
            self.episode_count = episode_count
            episode_steps, state, done = self.generate_episode(self.episode_count)    
            final_reward, modelling_outcomes = (self.delphos_interaction(state) if done else 0) or 0
            episode_rewards.append(final_reward)
            final_repr = self.state_manager.encode_state_to_string(state)
            self.model_set.add(final_repr)   
            
            L = len(episode_steps)
            
            if self.reward_distribution == 'uniform': # Constant reward: r_l = final_reward/L
                inter_reward = final_reward / L if L > 0 else final_reward
                for step_state, step_action, step_next_state in episode_steps:
                    self.replay_buffer.add((self.state_manager.encode_state_to_vector(step_state), step_action, inter_reward, self.state_manager.encode_state_to_vector(step_next_state), done))
                    self.buffer_log.append({'episode': episode_count, 'state': step_state, 'action': step_action, 'reward': inter_reward, 'next_state': step_next_state, 'done': done})
            elif self.reward_distribution == 'linear': # Linear reward: r_l = final_reward (l+1) / L                    
                for l, (step_state, step_action, step_next_state) in enumerate(episode_steps):
                    inter_reward = (final_reward * (l + 1)) / L if L > 0 else final_reward
                    self.replay_buffer.add((self.state_manager.encode_state_to_vector(step_state), step_action, inter_reward, self.state_manager.encode_state_to_vector(step_next_state), done))
                    self.buffer_log.append({'episode': episode_count, 'state': step_state, 'action': step_action, 'reward': inter_reward, 'next_state': step_next_state, 'done': done})
            else: # Discounted reward: r_l = γ^(L - l - 1) * final_reward
                gamma = self.discount_factor
                for l, (step_state, step_action, step_next_state) in enumerate(episode_steps):
                    discounted_reward = (gamma ** (L - l - 1)) * final_reward if L > 0 else final_reward
                    self.replay_buffer.add((self.state_manager.encode_state_to_vector(step_state),step_action,discounted_reward,self.state_manager.encode_state_to_vector(step_next_state),done))
                    self.buffer_log.append({'episode': episode_count, 'state': step_state, 'action': step_action, 'reward': discounted_reward, 'next_state': step_next_state, 'done': done})

            log = {'episode': episode_count, 'specification': final_repr, 'reward': final_reward, 'epsilon': self.epsilon}
            for key, value in modelling_outcomes.items():
                if key in self.reward_weights:
                    # If it's a Series (from .loc or filtering), extract the scalar
                    if isinstance(value, pd.Series):
                        log[key] = value.iloc[0] if not value.empty else None
                    else:
                        log[key] = value
            self.training_log.append(log)

            # Update best candidate tracker and apply experience replay
            metrics_dict = {}
            for key in modelling_outcomes.columns:
                value = modelling_outcomes[key]
                if isinstance(value, pd.Series):
                    metrics_dict[key] = value.iloc[0] if not value.empty else None
                else:
                    metrics_dict[key] = value
            self.update_candidate_tracker(episode_count, metrics_dict, state)
            self.perform_experience_replay()      
            
            if episode_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Epsilon decay per episode
            if episode_count < self.num_episodes * self.min_percentage:
                self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = 0.01

            # Early stopping based on rolling mean improvement.
            if episode_count >= self.min_episodes_before_stop:
                if self.check_early_stopping(episode_rewards):
                    logger.info("Early stopping condition met at episode %d.", episode_count)
                    early_stop_triggered = True
                    break
        training_time = time.time() - start_time
        total_episodes = episode_count + 1 if not early_stop_triggered else episode_count + 1
        self.plot_reward_learning_curve()
        self.save_training_logs(training_time, total_episodes, episode_rewards)

        # Update the rewards file with the new results
        results_file = os.path.join(self.path_rewards, "outputs", "rewards.csv")
        if os.path.exists(results_file):
            self.rewards = pd.read_csv(results_file)
        else:
            self.rewards = pd.DataFrame(columns=["specification"])

        job_results_file = os.path.join(self.path_rewards, "outputs", f"rewards_{self.agent_index}.csv")
        if os.path.exists(job_results_file):
            job_results = pd.read_csv(job_results_file)
            job_results = job_results[~job_results['specification'].isin(self.rewards['specification'])]
            self.rewards = pd.concat([self.rewards, job_results], ignore_index=True)
            self.rewards.to_csv(results_file, index=False)
            logger.info("Reward file update with new results, saved to %s", results_file)
        else:
            self.rewards.to_csv(results_file, index=False)
            logger.info("Reward file created, saved to %s", results_file)

        # delete the job_results_file
        if os.path.exists(job_results_file):
            os.remove(job_results_file)
            logger.info("Deleted job results file %s", job_results_file)


    def plot_reward_learning_curve(self) -> None:
        """
        Plot the learning curve of the agent using a rolling mean of training rewards.

        Notes:
            - The function uses Seaborn to create a line plot of the rolling mean of rewards.
            - The rolling mean is calculated over a window of episodes.
            - The plot is saved to the specified subfolder.
        """
        rewards = pd.DataFrame(self.training_log)
        if rewards.empty or 'episode' not in rewards.columns or 'reward' not in rewards.columns:
            logger.warning("Training log data unavailable or incomplete for plotting.")
            return

        rewards['mean_reward'] = rewards['reward'].rolling(window=200).mean()
        sns.set(style="whitegrid", palette="muted")
        ax = sns.lineplot(x='episode', y='mean_reward', data=rewards, label="Mean Reward", color='blue')

        ax.set_xlabel('Episodes')
        ax.set_ylabel(self.metric)
        ax.set_title('Learning curve')
        ax.grid(False)

        plot_path = os.path.join(self.subfolder, 'learning_curve.png')
        
        fig = ax.get_figure()
        fig.savefig(plot_path)
        fig.clf()

        logger.info("Learning curve saved to %s", self.subfolder + '/' + 'learning_curve.png')
    
    def save_training_logs(self, training_time: float, total_episodes: int, episode_rewards: List[float], action_log_path="action_log.csv", training_log_path="training_log.csv", buffer_log_path="buffer_log.csv") -> None:
        """
        Save the action log, buffer log, and training log to CSV files and write training metrics.

        Args:
            training_time (float): Total training time in seconds.
            total_episodes (int): Total number of episodes completed.
            episode_rewards (List[float]): List of rewards obtained during training.
            action_log_path (str): Path to save the action log CSV file.
            training_log_path (str): Path to save the training log CSV file.
            buffer_log_path (str): Path to save the buffer log CSV file.

        Notes:
            - The function saves the action log, buffer log, and training log as CSV files in the specified subfolder.
            - It also writes training metrics to a text file in the same subfolder.
            - The metrics include total episodes, training time, average reward, unique models explored, and converged models.
        """            

        pd.DataFrame(self.action_log).to_csv(f"{self.subfolder}/{action_log_path}", index=False)
        pd.DataFrame(self.buffer_log).to_csv(f"{self.subfolder}/{buffer_log_path}", index=False)
        pd.DataFrame(self.training_log).to_csv(f"{self.subfolder}/{training_log_path}", index=False)
        torch.save(self.policy_net.state_dict(), f"{self.subfolder}/dqn_model.pth" )   

        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        converged_models = len([r for r in episode_rewards if r > 0])
        unique_models = len(self.model_set)
        
        str_best_candidate = ""
        for metric in self.reward_weights:
            str_best_candidate += f"\nMetric: {metric}\n"
            for candidate in self.current_best_history:
                episode = candidate.get("episode")
                repr_key = f"{metric}_repr"
                value_key = f"{metric}_value"
                if repr_key in candidate and value_key in candidate:
                    repr_str = candidate[repr_key]
                    val = candidate[value_key]
                    if repr_str is not None and val is not None:
                        str_best_candidate += f"  Episode: {episode}, Value: {val:.4f}, Representation: {repr_str}\n"
                    
        metrics = (
            f"Total Episodes: {total_episodes}\n"
            f"Training Time (seconds): {training_time:.2f}\n"
            f"Average Reward: {avg_reward:.4f}\n"
            f"Unique Models Explored: {unique_models}\n"
            f"Converged Models: {converged_models}\n"
        )
        metrics += f"\nOverall Best Candidate History:\n{str_best_candidate}"
        metrics_path = os.path.join(self.subfolder, "training_metrics.txt")

        with open(metrics_path, "w") as f:
            f.write(metrics)
        logger.info("Training metrics saved to %s", metrics_path)
        logger.info("Training time: %.2f seconds", training_time)
        #logger.info(metrics)


########################################################################################
# Post-Training Utilities: AgentAnalyzer, BestCandidateEstimator
########################################################################################

class AgentAnalyzer:
    """
    Analyze the training outcomes and exploration of a DQN agent.
    Args:
        agent: An instance of a trained DQNLearner.
    """
    def __init__(self, agent) -> None:
        self.agent = agent

    def get_action_frequency_matrix(self) -> np.ndarray:
        """
        Compute a normalized frequency matrix for the agent's actions aggregated by batch.

        Returns:
            np.ndarray: 2D array (num_actions, num_batches) where each column sums to 1.
        """
        batch_size = self.agent.batch_size
        episodes = [entry["episode"] for entry in self.agent.action_log]
        if not episodes:
            return np.array([])
        max_episode = max(episodes)
        num_batches = math.ceil((max_episode + 1) / batch_size)
        num_actions = len(self.agent.action_to_index)
        freq_matrix = np.zeros((num_actions, num_batches))
        for entry in self.agent.action_log:
            ep = entry["episode"]
            batch_idx = ep // batch_size
            act = entry["action"]
            if isinstance(act, tuple):
                action_idx = self.agent.action_to_index.get(act)
            else:
                action_idx = int(act)
            if action_idx is not None:
                freq_matrix[action_idx, batch_idx] += 1
        for j in range(freq_matrix.shape[1]):
            col_sum = freq_matrix[:, j].sum()
            if col_sum > 0:
                freq_matrix[:, j] /= col_sum
        return freq_matrix

    def plot_q_distribution(self, save_path: str = "q_values_distribution.png") -> None:
        """
        Plot a heatmap of the action probability distribution over batches.

        Args:
            save_path (str): File path to save the plot.
        """
        freq_matrix = self.get_action_frequency_matrix()
        if freq_matrix.size == 0:
            logger.warning("No action log data available for heatmap.")
            return
        plt.figure(figsize=(12, 8))
        sns.heatmap(freq_matrix, cmap="viridis", cbar_kws={'label': 'Action probability'},
                    xticklabels=True, yticklabels=True)
        plt.xlabel("Batch (aggregated episodes)")
        plt.ylabel("Action index")
        plt.title("Action probability heatmap")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info("Action probability heatmap saved to %s", save_path)

    def plot_action_entropy(self, save_path: str = "action_entropy.png") -> None:
        """
        Compute and plot the entropy of the action distribution for each batch.

        Args:
            save_path (str): File path to save the plot.
        """
        freq_matrix = self.get_action_frequency_matrix()
        if freq_matrix.size == 0:
            logger.warning("No action log data available for entropy calculation.")
            return
        entropy = -np.sum(freq_matrix * np.log(freq_matrix + 1e-8), axis=0)
        batches = np.arange(freq_matrix.shape[1])
        plt.figure(figsize=(10, 6))
        plt.plot(batches, entropy, marker="o", linestyle="-")
        plt.xlabel("Batch")
        plt.ylabel("Entropy")
        plt.title("Action entropy over batches")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info("Action entropy plot saved to %s", save_path)

    def plot_best_candidate_trajectory(self, save_dir: str = ".") -> None:
        """
        Plots the best candidate trajectory across modelling outcomes.

        Args:
            save_dir (str): Directory to save the plots.

        Notes:
            - The function generates individual plots for each metric in the best candidate history.
            - It also creates a grouped plot with shared y-axes for rho metrics, AIC/BIC, and LLout.
            - The plots are saved in the specified directory.

            Saves:
            - One plot per modelling outcome (e.g., best_candidate_trajectory_AIC.png)
            - Grouped plot with multiple y-axes for different modelling outcomes types:
                - McFadden's pesudo-R: left y-axis (positive quadrant)
                - AIC/BIC: right y-axis (positive quadrant)
                - LL: lower plot (negative quadrant)

        """
        if not self.agent.current_best_history:
            logger.warning("No candidate history available for best candidate trajectory plot.")
            return

        df = pd.DataFrame(self.agent.current_best_history)
        df.sort_values(by="episode", inplace=True)

        # Plot individual metric trajectories
        for metric in self.agent.reward_weights:
            value_col = f"{metric}_value"
            if value_col not in df.columns:
                logger.warning(f"{value_col} not found in best candidate history.")
                continue
            plt.figure(figsize=(10, 6))
            changed_df = df[df[value_col].diff().fillna(1) != 0]
            plt.plot(changed_df["episode"], changed_df[value_col], marker="o", linestyle="-", label=metric)
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.title(f"Best Candidate Trajectory ({metric})")
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"best_candidate_trajectory_{metric}.png"))
            plt.close()
            logger.info(f"Best candidate trajectory plot for {metric} saved to {save_dir}")

        # --- Grouped y-axes plot: Rho metrics (left), AIC/BIC (right), LLout (bottom) ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.agent.reward_weights)))
        handles = []
        labels = []

        # Prepare right y-axis for top plot
        ax_top_right = ax_top.twinx()

        # Classify metrics
        rho_metrics = []
        aicbic_metrics = []
        llout_metrics = []
        for metric in self.agent.reward_weights.keys():
            m_lower = metric.lower()
            if "llout" in m_lower:
                llout_metrics.append(metric)
            elif "rho" in m_lower:
                rho_metrics.append(metric)
            elif "aic" in m_lower or "bic" in m_lower:
                aicbic_metrics.append(metric)
            else:
                # fallback: treat as rho
                rho_metrics.append(metric)

        # Plot Rho metrics on ax_top (left y-axis)
        for i, metric in enumerate(rho_metrics):
            color = colors[i % len(colors)]
            value_col = f"{metric}_value"
            if value_col not in df.columns:
                continue
            changed_df = df[df[value_col].diff().fillna(1) != 0].copy()
            line, = ax_top.plot(changed_df["episode"], changed_df[value_col], marker="o", linestyle="--", label=metric, color=color)
            handles.append(line)
            labels.append(metric)

        # Plot AIC/BIC metrics on ax_top_right (right y-axis)
        for j, metric in enumerate(aicbic_metrics):
            color = colors[(len(rho_metrics) + j) % len(colors)]
            value_col = f"{metric}_value"
            if value_col not in df.columns:
                continue
            changed_df = df[df[value_col].diff().fillna(1) != 0].copy()
            line, = ax_top_right.plot(changed_df["episode"], changed_df[value_col], marker="s", linestyle="--", label=metric, color=color)
            handles.append(line)
            labels.append(metric)

        # Plot LLout metrics on ax_bottom
        for k, metric in enumerate(llout_metrics):
            color = colors[(len(rho_metrics) + len(aicbic_metrics) + k) % len(colors)]
            value_col = f"{metric}_value"
            if value_col not in df.columns:
                continue
            changed_df = df[df[value_col].diff().fillna(1) != 0].copy()
            line, = ax_bottom.plot(changed_df["episode"], changed_df[value_col], marker="^", linestyle="--", label=metric, color=color)
            handles.append(line)
            labels.append(metric)

        # Set y-labels (only for axes that have lines)
        if rho_metrics:
            ax_top.set_ylabel("(Adj.) rho square")
        else:
            ax_top.set_ylabel("")
        if aicbic_metrics:
            ax_top_right.set_ylabel("AIC/BIC")
        else:
            ax_top_right.set_ylabel("")
        if llout_metrics:
            ax_bottom.set_ylabel("Log-likelihood")
        else:
            ax_bottom.set_ylabel("")

        ax_bottom.set_xlabel("Episode")
        ax_top.set_title("Best candidate trajectory")
        ax_top.grid(False)
        ax_bottom.grid(False)
        ax_top_right.grid(False)

        # Move legend outside
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_candidate_all.png"), bbox_inches="tight")
        plt.close()
        logger.info("All best candidate trajectory plot saved.")


    def plot_model_space_exploration(self, method: str = "PCA", save_path: str = "model_space_exploration.png") -> None:
        """
        Visualize the exploration of the model space by projecting state vectors to 2D.
        Args:
            method (str): Dimensionality reduction method ('PCA' or 'TSNE').
            save_path (str): File path for saving the scatter plot.
        """
        if not self.agent.current_best_history:
            logger.warning("No candidate history available for model space exploration plot.")
            return
        representations = [entry["representation"] for entry in self.agent.current_best_history]
        rewards = [entry["reward"] for entry in self.agent.current_best_history]
        state_vectors = []
        for rep in representations:
            state = self.agent.state_manager.decode_string_to_state(rep)
            state_vector = self.agent.state_manager.encode_state_to_vector(state).cpu().numpy()
            state_vectors.append(state_vector)
        state_vectors = np.array(state_vectors)
        if method.upper() == "PCA":
            reducer = PCA(n_components=2)
        elif method.upper() == "TSNE":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            logger.error("Unknown reduction method: %s. Use 'PCA' or 'TSNE'.", method)
            return
        projection = reducer.fit_transform(state_vectors)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=rewards, cmap="viridis", s=100, alpha=0.8)
        plt.colorbar(scatter, label="Reward")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"Model space exploration ({method.upper()})")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info("Model space exploration plot saved to %s", save_path)


class BestCandidateEstimator:
    
    def __init__(self, agent, results_file: str = 'rewards_best_candidates.csv'):
        
        self.agent = agent             
        self.job_output_dir = os.path.join(self.agent.path_rewards, "best_candidates")
        os.makedirs(self.job_output_dir, exist_ok=True)        


    def estimate_best_candidates(self, agent_index: int = 1, results: bool = False) -> None:
       
        from delphos import create_apollo_fixed, create_apollo_non_fixed, mnl_interaction

        unique_specs = set(entry["representation"] for entry in self.agent.current_best_history)      
        self.best_canidates = pd.DataFrame()

        for spec in unique_specs:           

            state_0, specific_0, covariates_0 = self.agent.state_manager.decode_string_to_specification(spec)
            specification = [f"{s}{sp}{c}" for s, sp, c in zip(state_0, specific_0, covariates_0)]
            model_name = "_".join(specification)

            print(f"\nEstimating model: {model_name}")

            apollo_beta_fixed = create_apollo_fixed(state_0, specific_0, covariates_0, self.agent.attributes, self.agent.covariates)
            apollo_beta_estimated = create_apollo_non_fixed(apollo_beta_fixed, self.agent.attributes, self.agent.covariates)
            outcome_df = mnl_interaction(model_name,apollo_beta_fixed,apollo_beta_estimated,self.agent.attributes, self.agent.covariates, agent_index, self.agent.path_rewards, self.agent.path_choice_dataset, False, True)
            new_df = pd.DataFrame(outcome_df)
            self.best_canidates = pd.concat([self.best_canidates, new_df], ignore_index=True)
            
        
        if results:
            self.outcomes_df.to_csv(self.job_results_file, index=False)
            return self.best_canidates
        
        print(f"\nAll estimations completed. Results saved to {self.job_results_file}")

#########################################################################################
# Specification Generation
#########################################################################################

import random
import pandas as pd

def generate_specification(num_var=48, fixed_middle=[1, 1], max_specs=200000, seed=42):
    random.seed(seed)

    num_middle = len(fixed_middle)
    num_comb = num_var - num_middle  
    specifications = set()

    while len(specifications) < max_specs:
        comb = [random.choice([0, 1]) for _ in range(num_comb)]
        spec = tuple([comb[0]] + fixed_middle + comb[1:])
        specifications.add(spec)
    return [list(s) for s in specifications]

# ========================================================================
#  Pareto class front plotting
# ========================================================================

class ParetoFrontAnalyzer:
    """
    Utility class for post-hoc analysis of model candidates and outcomes.
    """
    def plot_pareto_front(training_log_path, reward_path, output_path, metric_col='LLout'):
        rewards = pd.read_csv(reward_path)
        df = pd.read_csv(training_log_path)

        df['successfulEstimation'] = df['specification'].apply(
                lambda x: rewards[rewards['specification'] == x]['successfulEstimation'].values[0]
                if not rewards[rewards['specification'] == x].empty else None
                )


        df = df[df['successfulEstimation'] == True]
        df = df.dropna(subset=['numParams', metric_col])

                # Invert LLout for visualisation
        df['negLLout'] = -df[metric_col]

        # Identify Pareto front (min params, min negLLout → max LLout)
        df = df.sort_values(by=['numParams', 'negLLout'], ascending=[True, True])
        pareto = []
        best_y = float('inf') 
        for _, row in df.iterrows():
                y = row['negLLout']
                if y < best_y:
                        pareto.append(row)
                        best_y = y
        pareto_df = pd.DataFrame(pareto)

        # Identify best candidate (closest to origin by normalized distance)
        df['dist_to_origin'] = np.sqrt(df['numParams']**2 + df['negLLout']**2)
        best_model = df.loc[df['dist_to_origin'].idxmin()]
        worsest_model = df.loc[df['dist_to_origin'].idxmax()]

                # Plot
                # Sort Pareto front for plotting line correctly
        pareto_df = pareto_df.sort_values(by='numParams')

                # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df['numParams'], df['negLLout'], alpha=0.4, label='All Models')
        plt.plot(pareto_df['numParams'], pareto_df['negLLout'], color='red', marker='o', label='Pareto Front', linewidth=2)
        plt.xlabel("Number of Parameters")
        plt.ylabel(f"-{metric_col}")
        plt.title("Pareto front")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()
        plt.close()


