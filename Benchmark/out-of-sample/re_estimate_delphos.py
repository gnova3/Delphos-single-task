import sys
import os
from pathlib import Path
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DELPHOS_DIR = BENCHMARK_DIR / "Delphos"
sys.path.append(str(DELPHOS_DIR))

from apollo_new import apollo_probabilities
from specification_new import create_apollo_fixed, create_apollo_non_fixed

apollo = importr('apollo')

attributes = {
    1: [1, 2, 3],
    2: [1, 2, 3, 4],
    3: [1, 2]
}
covariates = {
    'age': [1, 2, 3, 4, 5], 
    'income': [1, 2, 3, 4],
    'class': [0, 1], 
    'ga': [0, 1], 
    'gender': [0, 1]
}

def evaluate_delphos_configs(configurations: dict[str, str], output_dir: Path):
    train_csv = str(BENCHMARK_DIR / "dataset" / "IS_swissmetro.csv")
    test_csv = str(BENCHMARK_DIR / "dataset" / "OOS_swissmetro.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = []
    
    for name, representation in configurations.items():
        print(f"\nEvaluating [{name}] configuration:\n{representation}")
        try:
            # 1. Parse the Delphos state array
            codes = representation.split('_')
            state = [int(c[0]) for c in codes]
            specific = [int(c[1]) for c in codes]
            covs = [int(c[2:]) for c in codes]

            apollo_beta_fixed = create_apollo_fixed(state, specific, covs, attributes, covariates, False)
            apollo_beta_estimated = create_apollo_non_fixed(apollo_beta_fixed, attributes, covariates)
            
            # Re-initialize Apollo to clear out old globals
            apollo.apollo_initialise()

            ro.globalenv['apollo_control']  = ro.ListVector({
                'modelName': f"re_estimate_{name}",  
                'modelDescr': f"Re-estimation for {name}", 
                'indivID': "ID", 
                'outputDirectory': str(output_dir)
            })
            
            # Load the strict benchmark datasets
            ro.globalenv['in_sample'] = ro.r['read.csv'](train_csv, header=True, sep=',')    
            ro.globalenv['out_of_sample'] = ro.r['read.csv'](test_csv, header=True, sep=',')    

            # Prepare initial starting values (beta)
            beta_names = create_apollo_fixed([0]*len(state), [0]*len(state), [0]*len(state), attributes, covariates, False)
            beta_values = []
            for parameter in beta_names:
                if parameter.startswith('L_'):
                    beta_values.append(0.1)
                else:
                    beta_values.append(0)
                    
            apollo_beta = ro.FloatVector(beta_values)
            apollo_beta.names = ro.StrVector(beta_names)

            ro.globalenv['apollo_beta'] = apollo_beta
            ro.globalenv['apollo_fixed'] = ro.StrVector(apollo_beta_fixed)

            # --- In-Sample Estimation ---
            ro.globalenv['database'] = ro.globalenv['in_sample']
            apollo_inputs = apollo.apollo_validateInputs()
            
            # This generates the `apollo_probabilities_function` internally in R
            apollo_probs = apollo_probabilities(apollo_beta, apollo_inputs, attributes, covariates, apollo_beta_estimated, False)    
            estimate_sett = ro.ListVector({'printLevel': 0, 'writeIter': False, 'silent': True})
            
            print("Estimating IS...")
            model = apollo.apollo_estimate(apollo_beta, ro.globalenv['apollo_fixed'], apollo_probs, apollo_inputs, estimate_sett)
            
            # Extract IS metrics
            is_aic = model.rx2('AIC')[0]
            is_ll = model.rx2('maximum')[0]
            num_params = model.rx2('nFreeParams')[0]
            
            # --- Out-of-Sample Prediction ---
            print("Simulating OOS...")
            ro.globalenv['model'] = model
            ro.globalenv['database'] = ro.globalenv['out_of_sample']
            
            # By executing functionality="estimate" on the OOS database, Apollo returns the probabilities
            # of the choices actually made by the OOS individuals. We sum their logs to get the LL.
            ro.r('''
                apollo_inputs_oos = apollo_validateInputs(silent = TRUE)
                P_oos = apollo_probabilities_function(model$estimate, apollo_inputs_oos, functionality="estimate")
                LL_oos = sum(log(P_oos))
            ''')
            
            oos_ll = ro.globalenv['LL_oos'][0]
            
            res = {
                "name": name,
                "configuration": representation,
                "IS_AIC": is_aic,
                "IS_log_likelihood": is_ll,
                "num_parameters": num_params,
                "OOS_log_likelihood": oos_ll
            }
            results_list.append(res)
            print(f"Success! IS_AIC: {is_aic:.2f}, IS_LL: {is_ll:.2f}, Params: {num_params}, OOS_LL: {oos_ll:.2f}")
            
        except Exception as e:
            print(f"Error evaluating [{name}]: {e}")
            results_list.append({"name": name, "configuration": representation, "error": str(e)})

    return results_list

if __name__ == "__main__":
    
    configs = {
        "best_Delphos": "101_313_215_312_204", 
        "worst_Delphos": "000_200_000_000_000"       
    }
    
    output_directory = BENCHMARK_DIR / "OOS" / "apollo_outputs"
    
    results = evaluate_delphos_configs(configs, output_directory)
    
    df_results = pd.DataFrame(results)
    output_csv = BENCHMARK_DIR / "OOS" / "re_estimation_delphos_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll Delphos configurations evaluated! Results saved to {output_csv}")
