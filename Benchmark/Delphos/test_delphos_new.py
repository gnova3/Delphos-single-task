"""Run with: python -m unittest discover -s Benchmark/Delphos -p 'test_*new.py'."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import torch
from delphos_new import DQNLearner
from results_new import ResultStore, BudgetExhausted
from specification_new import StateManager, create_apollo_fixed, create_apollo_non_fixed


class RefactorTests(unittest.TestCase):
    def test_state_encoding_and_action_modes(self):
        for tastes, covs in itertools.product([['generic'], ['generic', 'specific']], [[], ['age', 'income']]):
            manager = StateManager(dict(num_vars=3, taste=tastes, covariates=covs))
            actions, _ = manager.define_action_space()
            entries = [a[1:] for a in actions if a[0] == 'add']
            encodings = [tuple(manager.encode_state_to_vector([e]).tolist()) for e in entries]
            self.assertEqual(len(encodings), len(set(encodings)))
            for entry in entries:
                self.assertEqual(manager.decode_string_to_state(manager.encode_state_to_string([entry])), [entry])
            self.assertTrue(all(a[0] == 'add' for a in manager.mask_invalid_actions([], actions)))
            state = [entries[0]]
            for action in manager.mask_invalid_actions(state, actions):
                if action[0] == 'change':
                    self.assertEqual(action[1], entries[0][0])
                    self.assertNotEqual(action[1:], entries[0])

    def test_parameter_selection(self):
        attrs = {1: {1}, 2: {1, 2}, 3: {1}}
        covs = {'age': [1, 3], 'income': [0, 1]}
        fixed = create_apollo_fixed([1, 3, 2], [0, 0, 1], [1, 2, 0], attrs, covs)
        estimated = set(create_apollo_non_fixed(fixed, attrs, covs))
        self.assertEqual(estimated, {'asc_alt1_age_1', 'asc_alt1_age_3', 'asc_alt2_age_1', 'asc_alt2_age_3',
                                     'L_1', 'b_1_generic_box_cox_income_0', 'b_1_generic_box_cox_income_1', 'b_2_2_log'})
        self.assertEqual(len(fixed), len(set(fixed)))
        self.assertNotIn('L_0', fixed)

    def test_store_budget_cache_errors_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(Path(directory)/'models.sqlite', 2)
            def ok(name, before):
                before()
                return {'AIC': 10, 'successfulEstimation': True}
            first, cached = store.evaluate('100', ok)
            self.assertFalse(cached)
            self.assertTrue(store.evaluate('100', ok)[1])
            self.assertEqual(store.attempts, 1)
            def fail(name, before):
                before()
                raise ValueError('fit failed')
            row, _ = store.evaluate('200', fail)
            self.assertEqual(row['status'], 'error')
            self.assertEqual(store.attempts, 2)
            self.assertTrue(store.evaluate('200', fail)[1])
            with self.assertRaises(BudgetExhausted):
                store.evaluate('300', ok)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM models').fetchone()[0], 2)
            store.close()

    def make_agent(self, directory, budget=2, max_episodes=10):
        def estimator(name, before):
            before()
            return dict(successfulEstimation=True, AIC=10 if name.startswith('100') else 8, LL0=-20)
        return DQNLearner('unused.csv', Path(directory)/'run',
                           dict(num_vars=2, taste=['generic'], covariates=[], transformations=['linear']),
                           {1: {1}, 2: {1}}, {}, estimation_budget=budget, max_episodes=max_episodes,
                           estimator=estimator, batch_size=1, hidden_layers=(4,))

    def test_training_cache_budget_and_final_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory)
            states = iter([[(0, 'linear', 0, 0)], [(0, 'linear', 0, 0)], [(1, 'linear', 0, 0)]])
            def episode(number):
                state = next(states)
                action = agent.action_to_index[('add', *state[0])]
                return [([], action, state, False), (state, 0, state, True)], state, False
            agent.generate_episode = episode
            result = agent.train()
            self.assertEqual(result['estimations'], 2)
            self.assertEqual(result['episodes'], 3)
            self.assertEqual(result['cache_hits'], 1)
            self.assertEqual(result['best_candidates']['AIC']['value'], 8)
            self.assertEqual(result['stop_reason'], 'estimation_budget_reached')
            self.assertEqual([r[4] for r in agent.replay_buffer.buffer], [False, True]*3)
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy_net.parameters()))

    def test_zero_budget_and_external_stop(self):
        for budget, stop in [(0, None), (2, lambda: True)]:
            with tempfile.TemporaryDirectory() as directory:
                agent = self.make_agent(directory, budget)
                summary = agent.train(stop_requested=stop)
                self.assertEqual(summary['estimations'], 0)
                self.assertEqual(summary['episodes'], 0)
                self.assertEqual(summary['stop_reason'], 'external_stop' if stop else 'estimation_budget_reached')

    def test_episode_cap_and_preparation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, max_episodes=2)
            def fail(name, before):
                raise ValueError('preparation failed')
            agent.estimator = fail
            summary = agent.train()
            self.assertEqual(summary['estimations'], 0)
            self.assertEqual(summary['stop_reason'], 'episode_limit_before_budget')
            self.assertEqual(summary['successful_models'], 0)

    def test_episode_step_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory)
            agent.max_steps_per_episode = 2
            choices = iter([agent.action_to_index[('add', 0, 'linear', 0, 0)],
                            agent.action_to_index[('add', 1, 'linear', 0, 0)]])
            agent.select_action_index = lambda state: next(choices)
            steps, state, truncated = agent.generate_episode(0)
            self.assertTrue(truncated)
            self.assertEqual([s[-1] for s in steps], [False, True])


if __name__ == '__main__':
    unittest.main()
