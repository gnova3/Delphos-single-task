import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

from agent import *
pd.set_option('display.max_columns', None)

case                = 'decisions'   
agent_index         = 1
path_rewards        = 'dgp'
path_choice_dataset = 'decisions_data_sampled.csv'
path_to_save        = 'experiments/LL_params2'


state_space_params  = {'num_vars': 4, 
                        'transformations': ['linear', 'log', 'box-cox'], 'taste':['generic', 'specific'], 
                        'covariates': ['purpose', 'city', 'income_hh', 'female', 'age', 'education', 'hh_size', 'british_origin']}

attributes          = {1: {1,3}, 2: {1, 2, 3}, 3: {1, 2, 3}, 4: {1, 3}, 5: {1}, 6: {1}}
covariates          = {'purpose':[1, 2, 3, 4, 5, 6, 7, 8],
                       'city':[1,2,3],
                       'income_hh':[1,2,3,4,5,6,7,8,9,10,11,12],
                       'female':[0,1],
                       'age':[2,3,4,5,6,7,8,9],
                          'education':[1,2,3,4,5,6],
                            'hh_size':[1,2,3,4,5],
                            'british_origin':[0,1]}

reward_weights = {'LLout': 0.7, 'numParams':0.3}
behavioural_expectation = {1: '-', 3: '-'}

num_runs = 10

cfg = {
        "depth": 2,
        "hidden_units": 256,
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "total_episodes": 10_000,
        "buffer_size": 5_000,
        "target_update_freq": 5,
        "config_id": 1,
    }

if __name__ == '__main__':

    seed_list = [7*i + 1234 for i in range(num_runs)]
    
    if True:
        for i in range(num_runs):
                if i> 0:
                    continue

                       
                seed = seed_list[i]
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)                
                                   
                agent = DQNLearner(
                path_rewards,
                path_choice_dataset,
                path_to_save,
                state_space_params,
                cfg["total_episodes"],
                attributes,
                covariates,
                1,
                min_percentage=0.05,
                reward_weights=reward_weights,
                buffer_size=cfg["buffer_size"],
                target_update_freq=cfg["target_update_freq"],
                hidden_layers=[cfg["hidden_units"]] * cfg["depth"],
                learning_rate=cfg["learning_rate"],
                discount_factor=cfg["discount_factor"],
                behavioural_expectation=behavioural_expectation)


                agent.train()
                analyzer = AgentAnalyzer(agent)
                analyzer.plot_q_distribution(save_path=os.path.join(agent.subfolder, "q_values_distribution.png"))
                analyzer.plot_action_entropy(save_path=os.path.join(agent.subfolder, "action_entropy.png"))
                analyzer.plot_best_candidate_trajectory(save_dir=agent.subfolder)
                
