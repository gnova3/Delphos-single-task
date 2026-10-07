"""Delphos DQN learner: actions, episodes, rewards and training.

Apollo is loaded lazily so the learner can be tested with a lightweight estimator.
All outputs and cached models belong to this run alone.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from network import DQNetwork, ReplayBuffer
from results import ResultStore, successful
from specification import StateManager

logger = logging.getLogger(__name__)


class DQNLearner:
    def __init__(self, dataset, output_dir, state_space_params, attributes, covariates,
                 estimation_budget=None, max_episodes=10_000, max_steps_per_episode=100,
                 discount_factor=0.99, learning_rate=1e-4, buffer_size=5000,
                 batch_size=64, target_update_freq=5, hidden_layers=(256, 256),
                 reward_weights=None, behavioural_expectation=None,
                 reward_distribution='exponential', n_observations=None,
                 n_parameters_max=None, early_stop_window=500,
                 early_stop_tolerance=0.001, patience=100, min_percentage=0.5,
                 estimator=None):
        for name, value in [('max_episodes', max_episodes), ('max_steps_per_episode', max_steps_per_episode),
                            ('batch_size', batch_size), ('buffer_size', buffer_size),
                            ('target_update_freq', target_update_freq)]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 <= discount_factor <= 1:
            raise ValueError('discount_factor must be between 0 and 1')
        if reward_distribution not in ('uniform', 'linear', 'exponential'):
            raise ValueError('Unknown reward distribution')
        if n_observations is not None and (not isinstance(n_observations, int) or n_observations <= 0):
            raise ValueError('n_observations must be a positive integer or None')
        if n_parameters_max is not None and (not isinstance(n_parameters_max, int) or n_parameters_max <= 0):
            raise ValueError('n_parameters_max must be a positive integer or None')
        if not isinstance(early_stop_window, int) or early_stop_window <= 0:
            raise ValueError('early_stop_window must be a positive integer')
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError('patience must be a positive integer')
        if early_stop_tolerance < 0 or not 0 <= min_percentage <= 1:
            raise ValueError('early_stop_tolerance must be non-negative and min_percentage must be in [0, 1]')
        self.state_manager = StateManager(state_space_params)
        if self.state_manager.num_vars != max(max(v) for v in attributes.values()) + 1:
            raise ValueError('num_vars must equal the maximum attribute index plus one ASC')
        # Apollo and the agent must use exactly the same covariate code mapping.
        selected_covariates = state_space_params.get('covariates', [])
        self.covariates = {name: covariates[name] for name in selected_covariates}
        self.attributes = attributes
        self.behavioural_expectation = dict(behavioural_expectation or {})
        valid_attributes = set().union(*attributes.values())
        for attribute, expected_sign in self.behavioural_expectation.items():
            if attribute not in valid_attributes:
                raise ValueError(f'Behavioural expectation refers to unknown attribute {attribute}')
            if expected_sign not in ('+', '-'):
                raise ValueError("Behavioural expectations must use '+' or '-'")
        self.subfolder = Path(output_dir)
        self.subfolder.mkdir(parents=True, exist_ok=False)
        self.dataset = str(Path(dataset).resolve())
        if estimator is None:
            from apollo import ApolloEstimator
            estimator = ApolloEstimator(self.dataset, self.subfolder, attributes, self.covariates)
        self.estimator = estimator
        self.store = ResultStore(self.subfolder / 'models.sqlite', estimation_budget)
        self.estimation_budget = estimation_budget
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_distribution = reward_distribution
        self.reward_weights = reward_weights or {'AIC': 1.0}
        supported = {'AIC', 'BIC', 'LLout', 'rho2_0', 'adjRho2_0', 'rho2_C', 'adjRho2_C', 'numParams'}
        if not self.reward_weights.keys() <= supported:
            raise ValueError('Unsupported reward metric')
        self.LL0 = None
        self.Nobs = n_observations
        self.Kmax = n_parameters_max
        self.early_stop_window = early_stop_window
        self.early_stop_tolerance = early_stop_tolerance
        self.patience = patience
        self.min_episodes_before_stop = max_episodes * min_percentage
        self.no_improvement_count = 0
        self.best_candidates = {}
        self.current_best_history = []
        self.training_log = []
        self.action_log = []
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.action_space, self.action_to_index = self.state_manager.define_action_space()
        self.policy_net = DQNetwork(self.state_manager.get_state_length(), len(self.action_space), hidden_layers)
        self.target_net = DQNetwork(self.state_manager.get_state_length(), len(self.action_space), hidden_layers)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.epsilon = 1.0
        self.stop_reason = None
        self._trained = False
        self.config = dict(dataset=self.dataset, state_space_params=state_space_params,
                           attributes={str(k): sorted(v) for k, v in attributes.items()}, covariates=self.covariates,
                           estimation_budget=estimation_budget, max_episodes=max_episodes,
                           max_steps_per_episode=max_steps_per_episode, discount_factor=discount_factor,
                           learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size,
                           target_update_freq=target_update_freq, hidden_layers=list(hidden_layers),
                           reward_weights=self.reward_weights,
                           behavioural_expectation=self.behavioural_expectation,
                           reward_distribution=reward_distribution, n_observations=self.Nobs,
                           n_parameters_max=self.Kmax, early_stop_window=early_stop_window,
                           early_stop_tolerance=early_stop_tolerance, patience=patience,
                           min_percentage=min_percentage)
        (self.subfolder / 'config.json').write_text(json.dumps(self.config, indent=2))

    def valid_action_mask(self, state):
        mask = torch.zeros(len(self.action_space), dtype=torch.bool)
        for action in self.state_manager.mask_invalid_actions(state, self.action_space):
            mask[self.action_to_index[action]] = True
        return mask

    def select_action_index(self, state):
        mask = self.valid_action_mask(state)
        valid = mask.nonzero().flatten()
        if np.random.rand() < self.epsilon:
            return int(valid[np.random.randint(len(valid))])
        with torch.no_grad():
            return self.policy_net(self.state_manager.encode_state_to_vector(state)).masked_fill(~mask, -torch.inf).argmax().item()

    def apply_action(self, state, action_index):
        action = self.action_space[action_index]
        if action[0] == 'terminate':
            return list(state), True
        entry = tuple(action[1:])
        return sorted([s for s in state if s[0] != entry[0]] + [entry]), False

    def generate_episode(self, episode):
        state, steps = [], []
        for step in range(self.max_steps_per_episode):
            action = self.select_action_index(state)
            next_state, done = self.apply_action(state, action)
            # A finite horizon prevents cycles of change actions from hanging a run.
            terminal = done or step == self.max_steps_per_episode - 1
            steps.append((state, action, next_state, terminal))
            self.action_log.append(dict(episode=episode, state=state, action=self.action_space[action],
                                        next_state=next_state, done=terminal))
            state = next_state
            if terminal:
                return steps, state, not done

    def satisfies_behavioural_expectations(self, row):
        """Check the sign restrictions used by the previous DQNLearner.

        All generic and available alternative-specific coefficients for a
        constrained attribute are checked. Fixed coefficients are present in
        Apollo's output as zero and therefore satisfy either restriction.
        """
        for attribute, expected_sign in self.behavioural_expectation.items():
            generic_prefix = f'b_{attribute}_generic'
            specific_prefixes = [f'b_{alternative}_{attribute}' for alternative in self.attributes]
            matching_names = [
                name for name in row
                if name.startswith(generic_prefix)
                or any(name == prefix or name.startswith(f'{prefix}_') for prefix in specific_prefixes)
            ]
            for name in matching_names:
                try:
                    value = float(row[name])
                except (TypeError, ValueError):
                    logger.warning('Invalid coefficient %s for behavioural expectation', name)
                    return False
                if not np.isfinite(value):
                    logger.warning('Non-finite coefficient %s for behavioural expectation', name)
                    return False
                if (expected_sign == '-' and value > 0) or (expected_sign == '+' and value < 0):
                    return False
        return True

    def normalize_reward_metric(self, metric, value):
        """Return a fixed, baseline-relative reward per choice observation."""
        if (self.LL0 is None or self.Nobs is None or self.Nobs <= 0
                or value is None or not np.isfinite(value)):
            return 0.0
        if metric == 'LLout':
            normalized = (value - self.LL0) / self.Nobs
        elif metric == 'AIC':
            normalized = (-2 * self.LL0 - value) / self.Nobs
        elif metric == 'BIC':
            normalized = (-2 * self.LL0 - value) / self.Nobs
        elif metric == 'numParams':
            if self.Kmax is None:
                self.Kmax = 100
            normalized = (self.Kmax - value) / self.Kmax
        elif metric in ('rho2_0', 'adjRho2_0', 'rho2_C', 'adjRho2_C'):
            normalized = value
        else:
            raise ValueError(f'Unsupported reward metric: {metric}')
        if not np.isfinite(normalized):
            return 0.0
        return float(max(0.0, normalized))

    def reward_function(self, row):
        """Combine fixed-scale fit and complexity rewards using configured weights.

        For the Decisions benchmark this is
        ``0.7 * (LLout - LL0) / Nobs + 0.3 * (Kmax - numParams) / Kmax``.
        Each component is clipped at zero by ``normalize_reward_metric``.
        """
        if not successful(row):
            return 0.0
        if not self.satisfies_behavioural_expectations(row):
            return 0.0
        if row.get('LL0') is not None and np.isfinite(row['LL0']):
            self.LL0 = float(row['LL0'])
        if self.Nobs is None:
            degrees_of_freedom = row.get('numResids')
            parameters = row.get('numParams')
            if (degrees_of_freedom is not None and parameters is not None
                    and np.isfinite(degrees_of_freedom) and np.isfinite(parameters)):
                self.Nobs = int(degrees_of_freedom + parameters)
        reward = 0.0
        for metric, weight in self.reward_weights.items():
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            reward += weight * self.normalize_reward_metric(metric, value)
        return float(reward)

    def update_candidate_tracker(self, episode, row):
        if not successful(row):
            return
        for metric in self.reward_weights:
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            previous = self.best_candidates.get(metric)
            improves = previous is None or (value < previous['value'] if metric in ('AIC', 'BIC', 'numParams') else value > previous['value'])
            if improves:
                candidate = dict(metric=metric, value=value, episode=episode,
                                 estimation=self.store.attempts, specification=row['specification'])
                self.best_candidates[metric] = candidate
                self.current_best_history.append(candidate)
                logger.info('Best %s: %.4f at fit %d', metric, value, self.store.attempts)

    def perform_experience_replay(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        q = self.policy_net(torch.stack(states)).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(torch.stack(next_states)).masked_fill(~torch.stack(masks), -torch.inf).max(1).values
            next_q = torch.where(torch.tensor(dones), torch.zeros_like(next_q), next_q)
            targets = torch.tensor(rewards, dtype=torch.float32) + self.discount_factor * next_q
        loss = nn.functional.mse_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def check_early_stopping(self, episode_rewards):
        """Apply the convergence rule used in the previous DQNLearner."""
        if len(episode_rewards) < 2 * self.early_stop_window:
            return False
        current_mean = np.mean(episode_rewards[-self.early_stop_window:])
        previous_mean = np.mean(episode_rewards[-2 * self.early_stop_window:-self.early_stop_window])
        improvement = (current_mean - previous_mean) / (abs(previous_mean) + 1e-8)
        self.no_improvement_count = (self.no_improvement_count + 1
                                     if improvement < self.early_stop_tolerance else 0)
        return self.no_improvement_count >= self.patience

    def train(self, stop_requested=None):
        """Stop on reward convergence; episode cap is an explicitly reported safety limit.

        Optional stop_requested() supports an external cancellation signal.
        An optional estimation budget is retained for non-Decision experiments.
        """
        if self._trained:
            raise RuntimeError('Create a new learner for each run')
        self._trained = True
        started = time.perf_counter()
        try:
            self.stop_reason = 'episode_limit'
            episode_rewards = []
            for episode in range(self.max_episodes):
                if self.estimation_budget is not None and self.store.attempts >= self.estimation_budget:
                    self.stop_reason = 'estimation_budget_reached'
                    break
                if stop_requested is not None and stop_requested():
                    self.stop_reason = 'external_stop'
                    break
                steps, state, truncated = self.generate_episode(episode)
                if stop_requested is not None and stop_requested():
                    self.stop_reason = 'external_stop'
                    break
                name = self.state_manager.encode_state_to_string(state)
                row, cached = self.store.evaluate(name, self.estimator)
                reward = self.reward_function(row)
                episode_rewards.append(reward)
                length = len(steps)
                for i, (before, action, after, done) in enumerate(steps):
                    if self.reward_distribution == 'uniform':
                        step_reward = reward / length
                    elif self.reward_distribution == 'linear':
                        step_reward = reward * (i + 1) / length
                    else:
                        step_reward = reward * self.discount_factor ** (length - i - 1)
                    self.replay_buffer.add((self.state_manager.encode_state_to_vector(before), action, step_reward,
                                            self.state_manager.encode_state_to_vector(after), done, self.valid_action_mask(after)))
                self.training_log.append(dict(episode=episode, specification=name, reward=reward,
                                              epsilon=self.epsilon, estimations=self.store.attempts,
                                              cache_hit=cached, truncated=truncated, status=row['status'],
                                              **{m: row.get(m) for m in self.reward_weights}))
                self.update_candidate_tracker(episode, row)
                self.perform_experience_replay()
                if (episode + 1) % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.epsilon = max(0.01, 1.0 - (episode + 1) / self.max_episodes)
                if episode + 1 >= self.min_episodes_before_stop and self.check_early_stopping(episode_rewards):
                    self.stop_reason = 'reward_convergence'
                    break
                if self.estimation_budget is not None and self.store.attempts >= self.estimation_budget:
                    self.stop_reason = 'estimation_budget_reached'
                    break
        except KeyboardInterrupt:
            self.stop_reason = 'user_interrupted'
            raise
        except Exception:
            self.stop_reason = 'error'
            raise
        finally:
            try:
                self.save_results(time.perf_counter() - started)
            finally:
                self.store.close()
        return self.summary

    def save_results(self, seconds):
        self.config['n_observations'] = self.Nobs
        (self.subfolder / 'config.json').write_text(json.dumps(self.config, indent=2))
        self.store.export_csv(self.subfolder / 'models.csv')
        pd.DataFrame(self.training_log).to_csv(self.subfolder / 'training_log.csv', index=False)
        pd.DataFrame(self.action_log).to_csv(self.subfolder / 'action_log.csv', index=False)
        pd.DataFrame(self.current_best_history).to_csv(self.subfolder / 'best_history.csv', index=False)
        best_rows = [dict(metric=m, **self.store.cache[c['specification']]) for m, c in self.best_candidates.items()]
        pd.DataFrame(best_rows).to_csv(self.subfolder / 'best_candidates.csv', index=False)
        torch.save(self.policy_net.state_dict(), self.subfolder / 'dqn_model.pth')
        self.summary = dict(stop_reason=self.stop_reason, estimations=self.store.attempts,
                            budget=self.estimation_budget, episodes=len(self.training_log),
                            unique_specifications=len(self.store.cache),
                            successful_models=sum(successful(r) for r in self.store.cache.values()),
                            cache_hits=sum(r['cache_hit'] for r in self.training_log),
                            training_seconds=seconds, best_candidates=self.best_candidates)
        (self.subfolder / 'summary.json').write_text(json.dumps(self.summary, indent=2, allow_nan=False))
        logger.info('Run finished: %s (%d fits; budget=%s)',
                    self.stop_reason, self.store.attempts, self.estimation_budget)
