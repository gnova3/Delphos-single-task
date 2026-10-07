import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
VNS_DIR = BENCHMARK_DIR / "VNS"
sys.path.append(str(VNS_DIR))
from vns_benchmark_controlled import build_catalog, install_biogeme_331_parameter_workaround, aic_dimension
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.assisted import AssistedSpecification
from biogeme.catalog.specification import Specification

def re_estimate_and_predict(configurations: dict[str, str]):
    install_biogeme_331_parameter_workaround()
    
    train_path = BENCHMARK_DIR / "dataset" / "IS_swissmetro.csv"
    test_path = BENCHMARK_DIR / "dataset" / "OOS_swissmetro.csv"
    toml_path = VNS_DIR / "biogeme.toml"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    database, catalog, controller = build_catalog(train)
    model = BIOGEME(database, catalog, parameters=str(toml_path), generate_html=False, generate_yaml=False)
    model.model_name = "re_estimated_model"
    pareto_file = str(BENCHMARK_DIR / "OOS" / "dummy.pareto")
    assisted = AssistedSpecification(model, aic_dimension, pareto_file, parameter_file=str(toml_path))
    
    Specification.all_results = {}
    Specification.model_names = None
    
    results_list = []
    
    for name, config in configurations.items():
        print(f"\nEvaluating [{name}] configuration:\n{config}")
        try:
            controller.set_configuration_from_id(config)
            spec = Specification(configuration=controller.get_configuration(), biogeme_parameters=str(toml_path))
            est_results = spec.get_results()
            
            betas = est_results.get_beta_values()
            
            # Extract standard metrics from Biogeme results
            is_aic = est_results.akaike_information_criterion
            is_ll = est_results.final_loglikelihood
            k = est_results.number_of_parameters
            
            # 3. Simulate on OOS data
            test_db = Database('swissmetro_OOS', test)
            predictor = BIOGEME(test_db, {'log_probability': catalog.deep_flat_copy()}, parameters=str(toml_path), generate_html=False, generate_yaml=False)
            prediction = predictor.simulate(betas)['log_probability'].to_numpy()
            
            if len(prediction) != len(test) or not np.isfinite(prediction).all():
                raise ValueError('Non-finite or incomplete holdout predictions')
            
            # Calculate the Out-of-Sample log-likelihood
            oos_ll = float(prediction.sum())
            
            res = {
                "name": name,
                "configuration": config,
                "IS_AIC": is_aic,
                "IS_log_likelihood": is_ll,
                "num_parameters": k,
                "OOS_log_likelihood": oos_ll
            }
            results_list.append(res)
            print(f"Success! IS_AIC: {is_aic:.2f}, IS_LL: {is_ll:.2f}, Params: {k}, OOS_LL: {oos_ll:.2f}")
            
        except Exception as e:
            print(f"Error estimating [{name}]: {e}")
            results_list.append({"name": name, "configuration": config, "error": str(e)})
            
    return results_list

if __name__ == "__main__":
    
    best_sa_config = "asc:gender;b_cost:luggage;b_cost_gen_altspec:altspec;b_headway:age;b_headway_gen_altspec:altspec;b_seat:income;b_seat_gen_altspec:altspec;b_time:ga;b_time_gen_altspec:generic;choice_model:mnl;cost_form_alt1:log1p;cost_inclusion_alt1:on;headway_form_alt1:linear;headway_inclusion_alt1:on;seat_form_alt2:linear;seat_inclusion_alt2:on;time_form_alt1:log1p;time_inclusion_alt1:on"
    worst_sa_config = "asc:no_seg;b_cost:no_seg;b_cost_gen_altspec:generic;b_headway:no_seg;b_headway_gen_altspec:generic;b_seat:no_seg;b_seat_gen_altspec:generic;b_time:no_seg;b_time_gen_altspec:generic;choice_model:mnl;cost_form_alt1:linear;cost_inclusion_alt1:off;headway_form_alt1:linear;headway_inclusion_alt1:off;seat_form_alt2:linear;seat_inclusion_alt2:off;time_form_alt1:linear;time_inclusion_alt1:off"
    best_vns_config = "asc:no_seg;b_cost:class;b_cost_gen_altspec:altspec;b_headway:no_seg;b_headway_gen_altspec:generic;b_seat:no_seg;b_seat_gen_altspec:generic;b_time:who;b_time_gen_altspec:altspec;choice_model:mnl;cost_form_alt1:log1p;cost_inclusion_alt1:on;headway_form_alt1:linear;headway_inclusion_alt1:on;seat_form_alt2:linear;seat_inclusion_alt2:on;time_form_alt1:linear;time_inclusion_alt1:on"
    worst_vns_config = "asc:no_seg;b_cost:no_seg;b_cost_gen_altspec:generic;b_headway:no_seg;b_headway_gen_altspec:generic;b_seat:no_seg;b_seat_gen_altspec:generic;b_time:no_seg;b_time_gen_altspec:generic;choice_model:mnl;cost_form_alt1:linear;cost_inclusion_alt1:off;headway_form_alt1:linear;headway_inclusion_alt1:off;seat_form_alt2:linear;seat_inclusion_alt2:off;time_form_alt1:linear;time_inclusion_alt1:off"
    
    
    configs = {
        "best_SA": best_sa_config,
        "worst_SA": worst_sa_config,
        "best_VNS": best_vns_config,
        "worst_VNS": worst_vns_config
    }
    
    results = re_estimate_and_predict(configs)
    
    df_results = pd.DataFrame(results)
    output_csv = BENCHMARK_DIR / "OOS" / "re_estimation_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll configurations evaluated! Results saved to {output_csv}")