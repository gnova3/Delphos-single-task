"""
VNS Benchmark: This file separates data, catalogue, search, stopping, and reporting. 
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import importlib.metadata
import inspect
import shutil
import sys
from contextlib import contextmanager

from biogeme.assisted import AssistedSpecification
from biogeme.biogeme import BIOGEME
from biogeme.catalog import Catalog, CentralController, generic_alt_specific_catalogs, segmentation_catalogs
from biogeme.database import Database
from biogeme.expressions import Beta, Numeric, Variable, log
from biogeme.models import boxcox, loglogit
from biogeme.multiobjectives import loglikelihood_dimension


def aic_dimension(results):
    """Return [AIC, n_params] for a 2-D Pareto front that minimises AIC."""
    if results.raw_estimation_results is None:
        return [float(np.finfo(np.float32).max), float(np.finfo(np.float32).max)]
    return [results.akaike_information_criterion, results.number_of_parameters]
    
from biogeme.version import get_version
from biogeme.catalog.specification import Specification
from biogeme_optimization.vns import vns
from utils import load_toml, validity, audit_data, sha256, write_json




SCRIPT_DIR = Path(__file__).resolve().parent
TOML_FILE = SCRIPT_DIR / "biogeme.toml"
# ASC segmentation times four attribute groups, each with inclusion, form,
# generic/alternative-specific coding, and segmentation choices.
EXPECTED_NOMINAL_CONFIGURATIONS = 8 * (2 * 3 * 2 * 8) ** 4
# Only dormant off-state choices are collapsed here. This is NOT a count of
# distinct utilities: binary seat transformations and seat taste coding overlap.
COLLAPSED_INACTIVE_CONFIGURATION_STATES = 8 * (1 + 3 * 2 * 8) ** 4

TOML = load_toml(TOML_FILE)


##########################################################################  
#                    Biogeme version fix
###########################################################################
def install_biogeme_331_parameter_workaround() -> None:
    """Make candidate specifications honor the assisted-parameter object."""
    
    if getattr(Specification, "_vns_benchmark_331_workaround", False):
        return

    original_init = Specification.__init__

    def init_with_parameters(self: Specification, configuration: Any, biogeme_parameters: Any | None = None) -> None:
        if biogeme_parameters is None:
            biogeme_parameters = type(self).biogeme_parameters
        original_init(self, configuration=configuration, biogeme_parameters=biogeme_parameters)
        
    Specification.__init__ = init_with_parameters  # type: ignore[method-assign]
    Specification._vns_benchmark_331_workaround = True  # type: ignore[attr-defined]

##########################################################################  
#                    Catalogue defintion
###########################################################################

def inclusion_catalog(catalog_name: str, on_expression: Any, controlled_by: Any | None = None) -> Catalog:
    """
    Create a group-level attribute inclusion decision.
    - off is first so that VNS starts from a constants-only null model.
    - The alternative-specific copies of a term share the same controller.
    """
    return Catalog.from_dict(catalog_name=catalog_name, dict_of_expressions={"off": Numeric(0), "on": on_expression}, controlled_by=controlled_by,)

##########################################################################  
#                     Building Catalog from data
###########################################################################
def build_catalog(data: pd.DataFrame):
    """Declare the entire common candidate space before searching."""
    
    data = data.copy()
    required_columns = {"avail_1", "avail_2", "avail_3", "choice",
                        "x_1_1", "x_1_2", "x_1_3",
                        "x_2_1", "x_2_2", "x_2_3", "x_2_4",
                        "x_3_1", "x_3_2",
                        "ga", "luggage", "gender", "who", "income", "age", "class"}

    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    else:
        print("This dataset is OK. not missing checking is needed")

    seat_values = set(data["x_2_4"].dropna().unique().tolist())
    if seat_values != {0, 1}:
        raise ValueError(
            "x_2_4 must contain both values 0 and 1. Observed values are "
            f"{sorted(seat_values)}."
        )
    else:
        print("This dataset is OK. The values in x_2_4 are not missing")

    expected_categories = {
        "ga": {0, 1},
        "luggage": {0, 1, 3},
        "gender": {0, 1},
        "who": {0, 1, 2, 3},
        "income": {1, 2, 3, 4},
        "age": {1, 2, 3, 4, 5},
        "class": {0, 1},
    }
    for column, expected in expected_categories.items():
        observed = set(data[column].dropna().unique().tolist())
        if observed != expected:
            raise ValueError(
                f"Unexpected categories for {column}: {sorted(observed)}; "
                f"expected {sorted(expected)}."
            )
        else:
            print(f"This dataset is OK. The values in {column} are as expected")

    database = Database("swissmetro_vns_benchmark", data)
    
    avail_1 = Variable("avail_1")
    avail_2 = Variable("avail_2")
    avail_3 = Variable("avail_3")
    choice = Variable("choice")

    x_1_1, x_2_1, x_3_1 = Variable("x_1_1"), Variable("x_2_1"), Variable("x_3_1")
    x_1_2, x_2_2, x_3_2 = Variable("x_1_2"), Variable("x_2_2"), Variable("x_3_2")
    x_1_3, x_2_3 = Variable("x_1_3"), Variable("x_2_3")
    x_2_4 = Variable("x_2_4")

    asc_1 = Beta("asc1", 0, None, None, 0)
    asc_2 = Beta("asc2", 0, None, None, 0)
    b_time = Beta("b_time_base", 0, None, 0, 0)
    b_cost = Beta("b_cost_base", 0, None, 0, 0)
    b_headway = Beta("b_headway_base", 0, None, 0, 0)
    b_seat = Beta("b_seat_base", 0, 0, None, 0)
    lambda_time = Beta("lambda_time", 1, -10, 10, 0)
    lambda_cost = Beta("lambda_cost", 1, -10, 10, 0)
    lambda_headway = Beta("lambda_headway", 1, -10, 10, 0)
    lambda_seat = Beta("lambda_seat", 1, -10, 10, 0)

    # Covariate segmentation
    segmentations = (
        database.generate_segmentation(variable="ga", mapping={0: "noGA", 1: "GA"}),
        database.generate_segmentation(variable="luggage", mapping={0: "no_lugg", 1: "one_lugg", 3: "several_lugg"},),
        database.generate_segmentation(variable="gender", mapping={0: "g1", 1: "g2"}),
        database.generate_segmentation(variable="who", mapping={0: "w0", 1: "w1", 2: "w2", 3: "w3"}),
        database.generate_segmentation(variable="income", mapping={1: "I1", 2: "I2", 3: "I3", 4: "I4"}),
        database.generate_segmentation(variable="age", mapping={1: "a1", 2: "a2", 3: "a3", 4: "a4", 5: "a5"}),
        database.generate_segmentation(variable="class", mapping={0: "c0", 1: "c1"}))

    # Taste structure
    asc_1_catalog, asc_2_catalog = segmentation_catalogs(generic_name="asc", beta_parameters=[asc_1, asc_2], potential_segmentations=segmentations, maximum_number=1,)
    
    (time_betas,) = generic_alt_specific_catalogs(generic_name="b_time", beta_parameters=[b_time], alternatives=("alt1", "alt2", "alt3"), potential_segmentations=segmentations, maximum_number=1,)
    (cost_betas,) = generic_alt_specific_catalogs(generic_name="b_cost", beta_parameters=[b_cost], alternatives=("alt1", "alt2", "alt3"), potential_segmentations=segmentations, maximum_number=1,)
    (headway_betas,) = generic_alt_specific_catalogs(generic_name="b_headway", beta_parameters=[b_headway], alternatives=("alt1", "alt2"), potential_segmentations=segmentations, maximum_number=1,)

    seat_generic, seat_specific = segmentation_catalogs(generic_name="b_seat", beta_parameters=[b_seat, Beta("b_seat_base_alt2", 0, 0, None, 0)], potential_segmentations=segmentations, maximum_number=1,)
    seat_beta = {"alt2": Catalog.from_dict("b_seat_gen_altspec", {"generic": seat_generic, "altspec": seat_specific})}
    
    # Transformations
    time_1 = Catalog.from_dict("time_form_alt1", {"linear": x_1_1, "boxcox": boxcox(x_1_1, lambda_time), "log1p": log(1 + x_1_1),})
    time_2 = Catalog.from_dict( "time_form_alt2",{"linear": x_2_1, "boxcox": boxcox(x_2_1, lambda_time), "log1p": log(1 + x_2_1), }, controlled_by=time_1.controlled_by)
    time_3 = Catalog.from_dict("time_form_alt3", {"linear": x_3_1, "boxcox": boxcox(x_3_1, lambda_time), "log1p": log(1 + x_3_1), }, controlled_by=time_1.controlled_by)
    
    cost_1 = Catalog.from_dict( "cost_form_alt1",{"linear": x_1_2,"boxcox": boxcox(x_1_2, lambda_cost), "log1p": log(1 + x_1_2), }, )
    cost_2 = Catalog.from_dict( "cost_form_alt2",{"linear": x_2_2,"boxcox": boxcox(x_2_2, lambda_cost), "log1p": log(1 + x_2_2), }, controlled_by=cost_1.controlled_by)
    cost_3 = Catalog.from_dict( "cost_form_alt3",{"linear": x_3_2,"boxcox": boxcox(x_3_2, lambda_cost), "log1p": log(1 + x_3_2), }, controlled_by=cost_1.controlled_by)

    headway_1 = Catalog.from_dict("headway_form_alt1", {"linear": x_1_3, "boxcox": boxcox(x_1_3, lambda_headway), "log1p": log(1 + x_1_3), }, )
    headway_2 = Catalog.from_dict("headway_form_alt2",{"linear": x_2_3,"boxcox": boxcox(x_2_3, lambda_headway), "log1p": log(1 + x_2_3), }, controlled_by=headway_1.controlled_by)
    seat_2 = Catalog.from_dict("seat_form_alt2",{"linear": x_2_4,"boxcox": boxcox(x_2_4, lambda_seat), "log1p": log(1 + x_2_4), })

    time_term_1 = inclusion_catalog("time_inclusion_alt1", time_betas["alt1"] * time_1)
    time_term_2 = inclusion_catalog("time_inclusion_alt2", time_betas["alt2"] * time_2, controlled_by=time_term_1.controlled_by)
    time_term_3 = inclusion_catalog("time_inclusion_alt3", time_betas["alt3"] * time_3, controlled_by=time_term_1.controlled_by)

    cost_term_1 = inclusion_catalog("cost_inclusion_alt1", cost_betas["alt1"] * cost_1)
    cost_term_2 = inclusion_catalog("cost_inclusion_alt2", cost_betas["alt2"] * cost_2, controlled_by=cost_term_1.controlled_by)
    cost_term_3 = inclusion_catalog("cost_inclusion_alt3", cost_betas["alt3"] * cost_3, controlled_by=cost_term_1.controlled_by)

    headway_term_1 = inclusion_catalog("headway_inclusion_alt1", headway_betas["alt1"] * headway_1)
    headway_term_2 = inclusion_catalog("headway_inclusion_alt2", headway_betas["alt2"] * headway_2, controlled_by=headway_term_1.controlled_by)
    
    seat_term_2 = inclusion_catalog("seat_inclusion_alt2", seat_beta["alt2"] * seat_2)

    # UTILITIES
    v_1 = asc_1_catalog + time_term_1 + cost_term_1 + headway_term_1
    v_2 = asc_2_catalog + time_term_2 + cost_term_2 + headway_term_2 + seat_term_2
    v_3 = time_term_3 + cost_term_3
    utilities = {1: v_1, 2: v_2, 3: v_3}
    availability = {1: avail_1, 2: avail_2, 3: avail_3}

    # KERNEL: multinomial logit only; there is no nest or random coefficient search.
    log_probability = loglogit(utilities, availability, choice)
    model_catalog = Catalog.from_dict("choice_model", {"mnl": log_probability})

    # SPACE CHECK: fail if accidental controller sharing changes the search space.
    controller = CentralController(expression=model_catalog)
    number_of_configurations = controller.number_of_configurations()
    if number_of_configurations != EXPECTED_NOMINAL_CONFIGURATIONS:
        raise RuntimeError(f'Unexpected catalogue size: {number_of_configurations}')
    print(f" Number of configurations: {number_of_configurations}")
    return database, model_catalog, controller


##########################################################################  
#                    Budgeting the Estimator
##########################################################################  
class BudgetReached(BaseException):
    """Control-flow signal that cannot be swallowed by an estimator exception handler."""


@contextmanager
def metered_estimator(budget: int, ledger: Path, controller: CentralController):
    """Instrument the original estimator in memory; never edit package source.

    Checking BEFORE the call prevents fit budget+1. The completed budget-th fit
    still returns to native VNS so that its acceptance/Pareto update can finish.
    Diagnostics evaluate the likelihood derivatives once at fixed fitted betas;
    they do not optimise parameters or consume another fit.
    """
    original = BIOGEME.quick_estimate
    state = {'attempts': 0, 'rows': []}

    def measured(model, *args, **kwargs):
        if state['attempts'] >= budget:
            raise BudgetReached()
        state['attempts'] += 1
        record = {'attempt': state['attempts'],
                  'configuration': controller.get_configuration().get_string_id(),
                  'status': 'started'}
        # Persist before fitting, including attempts interrupted by a process crash.
        with ledger.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        started = time.perf_counter()
        try:
            result = original(model, *args, **kwargs)
            # quick_estimate omits derivatives; the original validity policy needs them.
            derivatives = model.function_evaluator.evaluate(
                the_betas=result.get_beta_values(), gradient=True, hessian=True, bhhh=True)
            result.raw_estimation_results.gradient = np.asarray(derivatives.gradient).tolist()
            result.raw_estimation_results.hessian = np.asarray(derivatives.hessian).tolist()
            result.raw_estimation_results.bhhh = np.asarray(derivatives.bhhh).tolist()
            # Reconstruct to refresh the package's derivative-availability flags.
            result = type(result)(raw_estimation_results=result.raw_estimation_results)
            valid, reason = validity(result)
            record.update(status='completed', valid=valid, invalid_reason=reason,
                          log_likelihood=float(result.final_loglikelihood),
                          parameters=result.number_of_parameters,
                          aic=float(result.akaike_information_criterion),
                          bic=float(result.bayesian_information_criterion),
                          betas=result.get_beta_values())
            return result
        except Exception as exc:
            # A computational exception is charged and terminates this seed visibly;
            # it is not silently retried or converted to a successful model.
            record.update(status='error', error=repr(exc))
            raise
        finally:
            record['seconds'] = time.perf_counter() - started
            state['rows'].append(record)
            with ledger.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(f"FIT {state['attempts']}/{budget}: {record['status']}", flush=True)

    BIOGEME.quick_estimate = measured
    try:
        yield state
    finally:
        BIOGEME.quick_estimate = original


def run_seed(args) -> None:
    """Run one isolated native VNS search, then evaluate its IS-selected model."""
    if get_version() != '3.3.1':
        raise RuntimeError('This experiment requires the archived Biogeme 3.3.1 API')
    run_dir = args.output / f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    
    shutil.copy2(TOML_FILE, run_dir / 'biogeme.toml')
    os.chdir(run_dir)
    train, test, audit = audit_data(args.train, args.test)
    database, catalog, controller = build_catalog(train)
    controller.reset_selection()
    initial_id = controller.get_configuration().get_string_id()
    settings = TOML['AssistedSpecification']
    manifest = {
        'seed': args.seed, 'fit_budget': args.budget, 'data': audit,
        'initial_model': 'constants_only_null', 'initial_configuration': initial_id,
        'objectives': ['IS_AIC', 'number_of_parameters'],
        'reporting_selection': 'minimum_IS_AIC_among_valid_fitted_models',
        'OOS_metric': 'mean_negative_log_likelihood_at_IS_betas',
        'budget_policy': 'every actual fit counts; cache hits and fixed-beta evaluations do not',
        'post_search_refits': 0, 'native_maximum_attempts': args.search_attempts,
        'native_settings': settings, 'nominal_configurations': EXPECTED_NOMINAL_CONFIGURATIONS,
        'collapsed_inactive_configuration_states': COLLAPSED_INACTIVE_CONFIGURATION_STATES,
        'source_files': {},
        'python': sys.version,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['biogeme', 'biogeme-optimization', 'numpy', 'pandas', 'jax', 'scipy']},
        'toml_sha256': sha256(TOML_FILE),
        'benchmark_files': {},
    }
    
    source_dir = run_dir / 'package_source'
    source_dir.mkdir()
    benchmark_dir = run_dir / 'benchmark_source'
    benchmark_dir.mkdir()
    for name in ('vns_main.py', 'vns_benchmark_controlled.py', 'utils.py'):
        source = SCRIPT_DIR / name
        shutil.copy2(source, benchmark_dir / name)
        manifest['benchmark_files'][name] = sha256(source)
    for obj in (vns, AssistedSpecification, Specification, BIOGEME):
        source = Path(inspect.getsourcefile(obj)).resolve()
        shutil.copy2(source, source_dir / source.name)
        manifest['source_files'][str(source)] = sha256(source)
    write_json(run_dir / 'manifest.json', manifest)
    model = BIOGEME(database, catalog, parameters=str(run_dir / 'biogeme.toml'),
                    generate_html=False, generate_yaml=False)
    model.model_name = f'vns_seed_{args.seed}'
    assisted = AssistedSpecification(model, aic_dimension,
                                    str(run_dir / 'search.pareto'), validity=validity,
                                    parameter_file=str(run_dir / 'biogeme.toml'))
    # The compatibility wrapper only forwards the existing parameter object.
    install_biogeme_331_parameter_workaround()
    Specification.all_results = {}
    Specification.model_names = None
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    reason = 'native_search_stopped_before_budget'
    state = {'attempts': 0, 'rows': []}
    try:
        with metered_estimator(args.budget, run_dir / 'estimations.jsonl', assisted.central_controller) as state:
            initial = Specification.default_specification()
            if initial.get_results().number_of_parameters != 2:
                raise AssertionError('Initial model must contain only two unsegmented ASCs')
            vns(problem=assisted, first_solutions=[initial.get_element(aic_dimension)], pareto=assisted.pareto, number_of_neighbors=settings['number_of_neighbors'], maximum_attempts=args.search_attempts)
    except BudgetReached:
        reason = 'estimation_budget_reached'
    except KeyboardInterrupt:
        reason = 'user_interrupted'
        raise
    except Exception as exc:
        reason = 'error'
        write_json(run_dir / 'error.json', {'error': repr(exc)})
        raise
    finally:
        assisted.pareto.dump()
        write_json(run_dir / 'status.json', {'stop_reason': reason, 'estimations': state['attempts'], 'budget': args.budget, 'search_seconds': time.perf_counter() - started})
    
    rows = state['rows']
    pd.DataFrame([{k: v for k, v in row.items() if k != 'betas'} for row in rows]).to_csv(run_dir / 'estimations.csv', index=False)
    valid_rows = [row for row in rows if row.get('valid')]
    if not valid_rows:
        raise RuntimeError('No valid fitted model available for reporting')
    best = min(valid_rows, key=lambda row: (row['aic'], row['configuration']))
    assisted.central_controller.set_configuration_from_id(best['configuration'])
    
    predictor = BIOGEME(Database('swissmetro_OOS', test), {'log_probability': catalog.deep_flat_copy()}, parameters=str(run_dir / 'biogeme.toml'), generate_html=False, generate_yaml=False)
    prediction = predictor.simulate(best['betas'])['log_probability'].to_numpy()
    
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError('Non-finite or incomplete holdout predictions')
    
    summary = {'seed': args.seed, 'budget': args.budget, 'estimations': state['attempts'],
               'stop_reason': reason, 'valid_estimations': len(valid_rows),
               'pareto_models': len(assisted.pareto.pareto),
               'selected_IS_AIC': best['aic'], 'selected_IS_BIC': best['bic'],
               'selected_parameters': best['parameters'],
               'selected_IS_log_likelihood': best['log_likelihood'],
               'OOS_log_likelihood': float(prediction.sum()),
               'OOS_mean_negative_log_likelihood': float(-prediction.mean()),
               'selected_configuration': best['configuration'],
               'elapsed_seconds': time.perf_counter() - started}
    write_json(run_dir / 'selected_model.json', best)
    write_json(run_dir / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    raise SystemExit('Run vns_main.py to launch the experiment.')
