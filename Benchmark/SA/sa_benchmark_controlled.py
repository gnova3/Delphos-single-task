"""Full VNS catalogue adapter for SearchLibrium 0.0.152.

Unmodified package methods perform SA iteration, cooling, acceptance, archive
updates, MNL likelihood calculation, bounded optimisation, and fit statistics.
The explicitly documented extensions provide catalogue moves, shared Box-Cox
design columns/derivatives, validity checks, complete caches, and a fit budget.
No Biogeme modules are imported and no installed package files are changed.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import tomllib

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
import SearchLibrium
from SearchLibrium.multinomial_logit import MultinomialLogit
from SearchLibrium.search import Parameters, Solution
from SearchLibrium.siman import SA

from utils import (DOMAINS, NOMINAL_CONFIGURATIONS, VNS_REFERENCE_HASHES, audit_data, build_design,
                   configuration_id, json_safe, null_configuration, sha256,
                   validity, write_json)

SCRIPT_DIR = Path(__file__).resolve().parent
VERSION = '0.0.152'


class BudgetReached(BaseException):
    """Propagate through native exception handlers before an extra fit starts."""


class CatalogueMNL(MultinomialLogit):
    """Native MNL estimator with a parameter-dependent design-matrix adapter.

    A shared lambda may affect several coefficient columns. Rather than replace
    the estimator, build those columns and call the package's MNL likelihood;
    JAX differentiates through the transformation. The package's original
    scipy_bfgs_optimization L-BFGS-B branch enforces the supplied bounds.
    """

    def __init__(self, design, maxiter=2000, tolerance=6.055454452393343e-7):
        super().__init__()
        self.design = design
        count = len(design.names)
        n = len(design.choice)
        matrix = np.asarray(design.matrix(design.initial))
        # Every coefficient column is already alternative/category coded. Native
        # intercepts and native per-column transformations must therefore be off.
        self.setup(X=matrix.reshape(n*3, count), y=design.choice.reshape(-1),
                   varnames=design.names, alts=np.tile([1, 2, 3], n),
                   ids=np.repeat(np.arange(n), 3), isvars=[], transvars=[],
                   avail=design.availability.reshape(-1), base_alt=3,
                   fit_intercept=False, init_coeff=design.initial.copy(),
                   maxiter=maxiter, ftol=tolerance, gtol=tolerance,
                   return_grad=True, return_hess=True, method='L-BFGS-B',
                   l1_penalty=0., l2_penalty=0.)
        self.bounds = design.bounds
        self.pval_penalty = 0.
        if self.Kf != count or not np.allclose(self.X, matrix):
            raise RuntimeError('Native setup changed the catalogue design')
        fixed = np.ones(count, dtype=bool)
        transformed = np.zeros(count, dtype=bool)
        choices = jnp.asarray(design.choice)
        availability = jnp.asarray(design.availability)

        def objective(theta):
            # The likelihood is the ORIGINAL SearchLibrium implementation.
            return MultinomialLogit._jax_mnl_negloglik(
                theta, design.matrix(theta, jnp), choices, availability,
                fixed, transformed, count, 0)

        self._objective = objective
        self._value_grad = jax.jit(jax.value_and_grad(objective))

    def get_loglik_and_gradient(self, betas, X, y, weights, avail):
        """Provide full chain-rule derivatives to the native bounded optimiser."""
        self.total_fun_eval += 1
        value, gradient = self._value_grad(jnp.asarray(betas))
        self.gtol_res = float(np.linalg.norm(np.asarray(gradient), ord=np.inf))
        return float(value), np.asarray(gradient, dtype=float)

    def compute_probabilities(self, betas, X, avail, return_logsum=False):
        """Use native probabilities on the design evaluated at the fitted lambdas."""
        return super().compute_probabilities(
            betas, np.asarray(self.design.matrix(betas)), avail, return_logsum)

    def diagnostics(self):
        """Fixed-beta evaluations, not an additional model estimation."""
        value, gradient = self._value_grad(jnp.asarray(self.coeff_est))
        # Native likelihood clips unavailable probabilities at 1e-300. Direct
        # second autodifferentiation can produce NaNs through those inactive
        # log terms. Differentiate its finite first gradient numerically instead;
        # these fixed-beta calls never enter the optimiser or consume fit budget.
        hessian = approx_derivative(
            lambda theta: np.asarray(self._value_grad(jnp.asarray(theta))[1]),
            np.asarray(self.coeff_est), method='3-point')
        hessian = (hessian + hessian.T) / 2.
        return -float(value), np.asarray(gradient), hessian


class CatalogueSA(SA):
    """Retain the native SA loop; adapt its proposal/evaluation interfaces.

    Native SA draws 1–5 moves per proposal and calls perturb_asfeature for each.
    Here a move changes one catalogue controller, keeping shared decisions
    atomic. All native acceptance, cooling and archive methods are inherited.
    """

    def __init__(self, train, args, settings):
        self.train = train
        self.args = args
        self.settings = settings
        self.attempts = 0
        self.records = []
        self.ledger = args.run_dir / 'estimations.jsonl'
        n = len(train)
        # A single placeholder selects the native asfeature proposal interface.
        # Candidate matrices are generated lazily, not expanded into millions
        # of precomputed columns, and OOS is never passed to the solver.
        params = Parameters(
            criterions=[('aic', -1)],
            df=pd.DataFrame({'catalogue': np.ones(n*3)}),
            varnames=['catalogue'], asvarnames=['catalogue'], isvarnames=[],
            choices=np.eye(3)[train.choice.to_numpy()-1].reshape(-1),
            choice_set=[1, 2, 3], alt_var=np.tile([1, 2, 3], n),
            choice_id=np.repeat(np.arange(n), 3),
            avail=train[['avail_1', 'avail_2', 'avail_3']].to_numpy().reshape(-1),
            base_alt=3, models=['multinomial'], allow_random=False,
            allow_bcvars=False, allow_corvars=False, all_sig=False, verbose=False)
        # Parameters' whitelist omits parameter count and its constructor assumes
        # multiobjective runs require test data. Neither objective here uses OOS.
        # Initialise with the IS objective, then register the second metric on
        # this instance; the native SA loop accepts a general objective vector.
        params.criterions = [('aic', -1), ('number_of_parameters', -1)]
        params.nb_crit = 2
        initial = self.new_solution(null_configuration())
        super().__init__(params, initial,
                         (args.initial_temperature, args.final_temperature,
                          args.temperature_steps, args.neighbours),
                         idnum=args.seed, output_dir=str(args.run_dir / 'native_sa'))

    def create_dummy_column(self, asvars):
        """The catalogue supplies all dummy columns; suppress random preselection."""
        return list(asvars)

    def remove_redundant_asvars(self, asvars, transasvars, asvarnames):
        return list(asvars)

    def _precompute_correlations(self, *args, **kwargs):
        """No correlation-based pruning: VNS's information set stays available."""
        return None

    @staticmethod
    def new_solution(configuration):
        result = Solution(2)
        result['configuration'] = dict(configuration)
        result['hash'] = configuration_id(configuration)
        result['model_n'] = 'multinomial'
        return result

    def setup_signature(self, sol):
        return configuration_id(sol['configuration'])

    def apply_constraints(self, sol):
        # Every controller choice is legal by construction. Statistical validity
        # is checked after fitting, just as in the VNS wrapper.
        configuration_id(sol['configuration'])
        return sol

    def repair_solution_for_clarity(self, sol):
        # Native repair treats individual column names as independent features.
        # Applying it here could split an atomic segmentation/attribute group.
        return sol

    def perturb_asfeature(self, sol):
        configuration = dict(sol['configuration'])
        keys = [key for key in sorted(DOMAINS) if len(DOMAINS[key]) > 1]
        key = keys[np.random.randint(len(keys))]
        options = [value for value in DOMAINS[key] if value != configuration[key]]
        configuration[key] = options[np.random.randint(len(options))]
        return self.new_solution(configuration)

    def _event(self, record):
        with self.ledger.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(json_safe(record), allow_nan=False) + '\n')

    def evaluate_solution(self, sol, track_best=True):
        """Cache whole results and meter each actual native fit, including failures."""
        key = self.setup_signature(sol)
        self.explored_specs.add(key)
        if key in self.evaluated_solutions:
            self.cache_hits += 1
            cached = self.copy_solution(self.evaluated_solutions[key])
            return cached, cached['converged']
        if self.attempts >= self.args.budget:
            raise BudgetReached()
        design = build_design(self.train, sol['configuration'])
        if len(design.names) > self.settings['maximum_number_parameters']:
            raise RuntimeError('Catalogue exceeded the declared parameter ceiling')
        model = CatalogueMNL(design, self.settings['max_iterations'], self.settings['tolerance'])
        # Charge immediately BEFORE fit, never by counting SA proposals or steps.
        self.attempts += 1
        record = {'attempt': self.attempts, 'configuration': key, 'status': 'started'}
        self._event(record)
        started = time.perf_counter()
        try:
            model.fit()  # ORIGINAL SearchLibrium fit and bounded optimiser.
            ll, gradient, hessian = model.diagnostics()
            valid, reason = validity(model.converged, model.coeff_est, ll, gradient,
                                     hessian, design, self.settings['identification_threshold'])
            # With all penalties zero, native fit statistics must be standard MLE.
            if not np.isclose(ll, model.loglik, rtol=1.e-10, atol=1.e-7):
                raise RuntimeError('Native fitted log likelihood disagrees with diagnostic evaluation')
            record.update(status='completed', valid=valid, invalid_reason=reason,
                          parameters=len(design.names), log_likelihood=ll,
                          bic=model.bic, aic=model.aic,
                          betas=dict(zip(design.names, model.coeff_est)),
                          smallest_information_eigenvalue=np.linalg.eigvalsh(hessian).min())
            sol['converged'] = valid
            sol['asvars'] = design.names
            sol['coeff_names'] = design.names
            sol['coeff'] = model.coeff_est.copy()
            sol['loglik'], sol['bic'], sol['aic'] = ll, model.bic, model.aic
            sol['number_of_parameters'] = len(design.names)
            sol['obj'] = np.array([model.aic, len(design.names)])
            # No fitted JAX objects in caches: coefficients plus the full config
            # suffice for reporting/prediction and avoid retaining every matrix.
            sol['model'] = None
            self.converged += int(valid)
            self.not_converged += int(not valid)
            self.evaluated_solutions[key] = self.copy_solution(sol)
            return sol, valid
        except Exception as exc:
            record.update(status='error', valid=False, error=repr(exc))
            # As in the VNS driver, abort computational exceptions visibly.
            # BaseException bypasses native SA's catch-and-continue handlers.
            raise EstimationFailure(str(exc)) from exc
        finally:
            record['seconds'] = time.perf_counter() - started
            self.records.append(record)
            self._event(record)
            print(f'FIT {self.attempts}/{self.args.budget}: {record["status"]}', flush=True)

    def prepare_to_run(self):
        # Estimate the same null before native prepare_to_run; supplying it avoids
        # native random starting-model selection and temperature-calibration fits.
        self.current_sol, valid = self.evaluate_solution(self.current_sol)
        if not valid:
            raise EstimationFailure('Constants-only initial model failed validity')
        super().prepare_to_run()


class EstimationFailure(BaseException):
    """A failed charged fit terminates this seed, not the remaining batch."""


def read_settings():
    """Read numerical controls as data only; never import the running VNS code."""
    for name, expected in VNS_REFERENCE_HASHES.items():
        if sha256(SCRIPT_DIR.parent / 'VNS' / name) != expected:
            raise RuntimeError(f'VNS {name} changed since the catalogue audit. '
                               'Review catalogue parity before updating VNS_REFERENCE_HASHES.')
    path = SCRIPT_DIR.parent / 'VNS/biogeme.toml'
    with path.open('rb') as stream:
        declared = tomllib.load(stream)
    return {
        'maximum_number_parameters': declared['AssistedSpecification']['maximum_number_parameters'],
        'identification_threshold': declared['Output']['identification_threshold'],
        'max_iterations': declared['SimpleBounds']['max_iterations'],
        'tolerance': declared['SimpleBounds']['tolerance'],
        'VNS_settings_file': str(path), 'VNS_settings_sha256': sha256(path),
    }


def run_seed(args):
    if importlib.metadata.version('SearchLibrium') != VERSION:
        raise RuntimeError(f'This audited adapter requires SearchLibrium {VERSION}')
    args.run_dir = args.output / f'seed_{args.seed}'
    args.run_dir.mkdir(parents=True, exist_ok=False)
    train, test, audit = audit_data(args.train, args.test)
    settings = read_settings()
    source_dir = args.run_dir / 'package_source'
    source_dir.mkdir()
    package = Path(SearchLibrium.__file__).resolve().parent
    package_hashes = {}
    for source in sorted(package.glob('*.py')):
        shutil.copy2(source, source_dir / source.name)
        package_hashes[source.name] = sha256(source)
    benchmark_dir = args.run_dir / 'benchmark_source'
    benchmark_dir.mkdir()
    for name in ('utils.py', 'sa_main.py', 'sa_benchmark_controlled.py'):
        shutil.copy2(SCRIPT_DIR / name, benchmark_dir / name)
    manifest = {
        'seed': args.seed, 'fit_budget': args.budget, 'data': audit,
        'catalogue': DOMAINS, 'nominal_configurations': NOMINAL_CONFIGURATIONS,
        'initial_configuration': configuration_id(null_configuration()),
        'objectives': ['minimise_IS_AIC', 'minimise_parameter_count'],
        'selection': 'minimum_IS_AIC_among_all_valid_fits', 'post_search_refits': 0,
        'native_SA_controls': [args.initial_temperature, args.final_temperature,
                               args.temperature_steps, args.neighbours],
        'native_MLE_method': 'L-BFGS-B', 'l1_l2_pvalue_penalties': [0, 0, 0],
        'native_multiobjective_acceptance': 'dominance; dominated proposal accepted with probability 0.10',
        'information_matrix_method': 'three-point differentiation of native likelihood first gradient',
        'settings': settings, 'python': sys.version,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('SearchLibrium', 'numpy', 'pandas', 'scipy', 'jax')},
        'package_source_hashes': package_hashes,
        'benchmark_source_hashes': {p.name: sha256(p) for p in benchmark_dir.iterdir()},
        'VNS_source_hashes': {name: sha256(SCRIPT_DIR.parent / 'VNS' / name) for name in
                              ('vns_main.py', 'vns_benchmark_controlled.py', 'utils.py')},
        'adapter': 'catalogue moves, shared transformations and derivatives, bounds, validity, cache, budget',
        'budget_policy': 'actual fits including invalid/errors count; cached results and fixed-beta evaluation do not',
        'stopping_difference': 'SA cooling exhaustion is native; VNS neighbourhood exhaustion is native',
        'validation_scope': 'same declared utility space; different optimisers can give different convergence/results',
    }
    write_json(args.run_dir / 'manifest.json', manifest)
    if args.dry_run:
        write_json(args.run_dir / 'status.json', {'stop_reason': 'dry_run', 'estimations': 0})
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    solver = CatalogueSA(train, args, settings)
    started = time.perf_counter()
    stop_reason = 'native_SA_stopped_before_budget'
    error = None
    try:
        solver.run_search()  # ORIGINAL SearchLibrium SA loop.
    except BudgetReached:
        stop_reason = 'estimation_budget_reached'
    except KeyboardInterrupt:
        stop_reason = 'user_interrupted'
        raise
    except (EstimationFailure, Exception) as exc:
        stop_reason, error = 'error', repr(exc)
    finally:
        solver.close_files()
        if solver.attempts == args.budget and error is None and stop_reason != 'user_interrupted':
            stop_reason = 'estimation_budget_reached'
        write_json(args.run_dir / 'status.json', {
            'stop_reason': stop_reason, 'estimations': solver.attempts,
            'budget': args.budget, 'search_seconds': time.perf_counter()-started, 'error': error})
        pd.DataFrame([{k: v for k, v in r.items() if k != 'betas'}
                      for r in solver.records]).to_csv(args.run_dir / 'estimations.csv', index=False)
        write_json(args.run_dir / 'native_archive.json', [
            {'configuration': solver.setup_signature(sol), 'objectives': sol.get_obj()}
            for sol in solver.archive])
    if error:
        raise RuntimeError(error)
    candidates = [record for record in solver.records if record.get('valid')]
    if not candidates:
        raise RuntimeError('No valid reporting model')
    selected = min(candidates, key=lambda r: (r['aic'], r['configuration']))
    config = dict(part.split(':', 1) for part in selected['configuration'].split(';'))
    holdout = build_design(test, config)
    theta = np.array([selected['betas'][name] for name in holdout.names])
    # Fixed-beta native likelihood evaluation: no fitting or holdout selection.
    count = len(theta)
    oos_ll = -float(MultinomialLogit._jax_mnl_negloglik(
        jnp.asarray(theta), holdout.matrix(jnp.asarray(theta), jnp),
        jnp.asarray(holdout.choice), jnp.asarray(holdout.availability),
        np.ones(count, bool), np.zeros(count, bool), count, 0))
    if not np.isfinite(oos_ll):
        raise RuntimeError('Non-finite holdout likelihood')
    write_json(args.run_dir / 'selected_model.json', selected)
    write_json(args.run_dir / 'summary.json', {
        'seed': args.seed, 'budget': args.budget, 'estimations': solver.attempts,
        'valid_estimations': len(candidates), 'stop_reason': stop_reason,
        'cache_hits': solver.cache_hits, 'native_archive_models': len(solver.archive),
        'selected_IS_AIC': selected['aic'], 'selected_IS_BIC': selected['bic'],
        'selected_parameters': selected['parameters'],
        'selected_IS_log_likelihood': selected['log_likelihood'],
        'selected_configuration': selected['configuration'],
        'OOS_log_likelihood': oos_ll, 'OOS_mean_negative_log_likelihood': -oos_ll / len(test),
        'elapsed_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    raise SystemExit('Run sa_main.py to launch the experiment.')
