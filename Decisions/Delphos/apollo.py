"""Apollo/R likelihood and estimation, extracted from the original delphos.py.

The original 80/20 individual split (R seed 123), MNL probabilities, panel
product, starting values, and Apollo estimation settings are retained.
"""
import os
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from specification import create_apollo_fixed, create_apollo_non_fixed

apollo = importr('apollo')

def apollo_probabilities(apollo_beta, apollo_inputs, attributes, covariates, apollo_beta_estimated, r_print=False):
    
    num_alternatives = max(attributes.keys())

    ro.globalenv['apollo_beta'] = apollo_beta
    ro.globalenv['apollo_inputs'] = apollo_inputs
    ro.globalenv['apollo_estimated'] = apollo_beta_estimated

    utility_functions = []    
    for alt in range(1, num_alternatives + 1):
            terms = []
                
            # Alternative-Specific Constants
            asc_param = f'asc_alt{alt}'
            if asc_param in apollo_beta_estimated:
                terms.append(asc_param)   
            
            for cov in covariates:
                min_category = min(covariates[cov])
                max_category = max(covariates[cov])                
                asc_cov_param = f'asc_alt{alt}_{cov}_{min_category}'
                if asc_cov_param in apollo_beta_estimated:
                    interaction_terms = [f'asc_alt{alt}_{cov}_{k} * ({cov} == {k})' for k in covariates[cov]]
                    terms.append(f'({" + ".join(interaction_terms)})')

            # Attribute-specific Coefficients
            if alt in attributes:
                for attr in attributes[alt]:                    
                    # Generic terms
                    generic_terms = [
                        (f'b_{attr}_generic', f'x_{alt}_{attr}'),
                        (f'b_{attr}_generic_log', f'log(1 + x_{alt}_{attr})'),
                        (f'b_{attr}_generic_box_cox', f'(x_{alt}_{attr}**L_{attr} - 1)/L_{attr}')
                    ]
                    
                    for param, expr in generic_terms:
                        if param in apollo_beta_estimated:
                            terms.append(f'({param} * {expr})')

                    # Alternative-specific terms
                    alt_specific_terms = [
                        (f'b_{alt}_{attr}', f'x_{alt}_{attr}'),
                        (f'b_{alt}_{attr}_log', f'log(1 + x_{alt}_{attr})'),
                        #(f'b_{alt}_{attr}_box_cox', f'(x_{alt}_{attr}**L_{alt}_{attr} - 1)/L_{alt}_{attr}')
                        (f'b_{alt}_{attr}_box_cox', f'(x_{alt}_{attr}**L_{attr} - 1)/L_{attr}')
                    ]
                    for param, expr in alt_specific_terms:
                        if param in apollo_beta_estimated:
                            terms.append(f'({param} * {expr})')

                    # Interaction terms
                    for cov in covariates:
                        min_category = min(covariates[cov])
                        max_category = max(covariates[cov])  

                        generic_cov_terms = [(f'b_{attr}_generic_{cov}_{k}', f'x_{alt}_{attr} * ({cov} == {k})') for k in covariates[cov]] 
                        generic_cov_terms_log = [(f'b_{attr}_generic_log_{cov}_{k}', f'log(1 + x_{alt}_{attr}) * ({cov} == {k})') for k in covariates[cov]]
                        generic_cov_terms_box_cox = [(f'b_{attr}_generic_box_cox_{cov}_{k}', f'((x_{alt}_{attr}**L_{attr} - 1)/L_{attr}) * ({cov} == {k})') for k in covariates[cov]]
                        generic_cov_terms.extend(generic_cov_terms_log)
                        generic_cov_terms.extend(generic_cov_terms_box_cox)
                        generic_terms_to_add = []
                        for param, expr in generic_cov_terms:
                            if param in apollo_beta_estimated:
                                generic_terms_to_add.append(f"({param} * {expr})")
                        if generic_terms_to_add:
                            terms.append(f'({" + ".join(generic_terms_to_add)})')
                        
                        alt_cov_terms = [(f'b_{alt}_{attr}_{cov}_{k}', f'x_{alt}_{attr} * ({cov} == {k})') for k in covariates[cov]]
                        alt_cov_terms_log = [(f'b_{alt}_{attr}_log_{cov}_{k}', f'log(1 + x_{alt}_{attr}) * ({cov} == {k})') for k in covariates[cov]]
                        #alt_cov_terms_box_cox = [(f'b_{alt}_{attr}_box_cox_{cov}_{k}', f'((x_{alt}_{attr}**L_{alt}_{attr} - 1)/L_{alt}_{attr}) * ({cov} == {k})') for k in covariates[cov]]
                        alt_cov_terms_box_cox = [(f'b_{alt}_{attr}_box_cox_{cov}_{k}', f'((x_{alt}_{attr}**L_{attr} - 1)/L_{attr}) * ({cov} == {k})') for k in covariates[cov]]
                        alt_cov_terms.extend(alt_cov_terms_log)
                        alt_cov_terms.extend(alt_cov_terms_box_cox)
                        alt_terms_to_add = []
                        for param, expr in alt_cov_terms:
                            if param in apollo_beta_estimated:                    
                                alt_terms_to_add.append(f"({param} * {expr})")
                        if alt_terms_to_add:
                            terms.append(f'({" + ".join(alt_terms_to_add)})')
            if not terms:
                terms.append('0')

            utility_functions.append(f'V[["Alt{alt}"]] = ' + " + ".join(terms))
    

    utility_code = "\n        ".join(utility_functions)

    r_code = f"""
         apollo_probabilities_function <- function(apollo_beta, apollo_inputs, functionality = 'estimate') {{
            apollo_attach(apollo_beta, apollo_inputs)
            on.exit(apollo_detach(apollo_beta, apollo_inputs))
            
            P = list()
            V = list()
            
            {utility_code}
            
            mnl_settings = list(
                alternatives = c({", ".join([f"Alt{alt}={alt}" for alt in range(1, num_alternatives + 1)])}),
                avail = list({", ".join([f"Alt{alt}=avail_{alt}" for alt in range(1, num_alternatives + 1)])}),
                choiceVar = choice,
                utilities = V
            )
            
            P[["model"]] = apollo_mnl(mnl_settings, functionality)
            #P = apollo_panelProd(P, apollo_inputs, functionality)
            P = apollo_prepareProb(P, apollo_inputs, functionality)
            return(P)
        }}
        """
    if r_print:
        print(r_code)
    else:
        ro.r(r_code)
        return ro.globalenv['apollo_probabilities_function']

def mnl_interaction(name, apollo_beta_fixed, apollo_beta_estimated, attributes, covariates, job_index, path_dir='specification', df = 'train.csv', info=False, before_estimate=None):

    """
    Dynamically configures and runs an MNL interaction model using Apollo.

    Args:
        name: Name of the model.
        case: Specific case or directory name.
        beta_fixed: Fixed beta parameters for the model.
        num_alternatives: Number of alternatives in the choice model.
        num_attributes: Number of attributes per alternative.
        path_dir: Directory for model files and outputs.
    """
    num_alternatives = max(attributes.keys())
    num_attributes = max(max(attr_list) for attr_list in attributes.values())

    output_directory   = f'{path_dir}/outputs/job_{job_index}'
    database_directory = os.path.join(path_dir, df)    
    save_directory     = f'{path_dir}/outputs/job_{job_index}/{name}_results.csv'
    
    os.makedirs(output_directory, exist_ok=True)
    apollo.apollo_initialise()

    ro.globalenv['apollo_control']  = ro.ListVector({'modelName': f"{name}",  
                                                     'modelDescr': "MNL model on choice SP data", 
                                                     'indivID': "ID", 
                                                     'outputDirectory': f"{output_directory}"})
    
    
    ro.globalenv['database_1'] = ro.r['read.csv'](str(database_directory), header=True, sep=',')    

    ro.r('''    
        set.seed(123) 
        individuals <- unique(database_1$ID)
        n_individuals <- length(individuals)

        train_individuals <- sample(individuals, size = 0.8 * n_individuals)
        test_individuals  <- setdiff(individuals, train_individuals)
        database_1$test <- ifelse(database_1$ID %in% test_individuals, 1, 0)
        in_sample <- subset(database_1, test == 0)
         
        database  <- database_1                  
        ''')
    
    state_0 = [0] *  (num_attributes + 1)  # Initial state (assume only ASC enabled)
    specific_0 = [0] * (num_attributes + 1)  # Assume generic by default
    covariates_0 = [0] * (num_attributes + 1)  # No covariates initially

    beta_names = create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates, False)
    beta_values = []

    for parameter in beta_names:
        if parameter.startswith('L_'):
            beta_values.append(0.1)
        else:
            beta_values.append(0)
        
    apollo_beta                         = ro.FloatVector(beta_values)
    apollo_beta.names                   = ro.StrVector(beta_names)

    ro.globalenv['apollo_beta']         = apollo_beta
    ro.globalenv['apollo_fixed']        = ro.StrVector(apollo_beta_fixed)

    ro.globalenv['database']            = ro.globalenv['in_sample']
    apollo_inputs                       = apollo.apollo_validateInputs()
    apollo_probs                        = apollo_probabilities(apollo_beta, apollo_inputs, attributes, covariates, apollo_beta_estimated, False)    
    estimate_sett                       = ro.ListVector({'printLevel': 0,'writeIter': False,'silent': True})
    ro.globalenv['estimate_settings']   = estimate_sett
    


    if before_estimate is not None:
        before_estimate()
    model                               = apollo.apollo_estimate(apollo_beta, ro.globalenv['apollo_fixed'], apollo_probs, apollo_inputs, estimate_sett)
    if info:
        apollo.apollo_modelOutput(model)
    ro.globalenv['model'] = model
    ro.globalenv['save_path'] = f"{save_directory}"
    apollo.apollo_saveOutput(model)
    
    ro.r('''
        model_summary <- data.frame(
                numParams = model$numParams,
                numResids = model$numResids,
                maximum = model$maximum,
                vcHessianConditionNumber = model$vcHessianConditionNumber,
                successfulEstimation = model$successfulEstimation,
                LL0 = model$LL0,
                LLC = model$LLC,
                LLout = model$LLout,
                rho2_0 = model$rho2_0,
                adjRho2_0 = model$adjRho2_0,
                rho2_C = model$rho2_C,
                adjRho2_C = model$adjRho2_C,
                AIC = model$AIC,
                BIC = model$BIC,
                eigValue = model$eigValue,
                timeTaken = model$timeTaken,
                nFreeParams = model$nFreeParams,
                t(data.frame(value = model$estimate, row.names = names(model$estimate))),
                t(data.frame(value = c(model$se, model$robse), row.names = c(paste0('se_', names(model$se)), paste0('rob_', names(model$robse)))))
        )
             
        cat('Saving model summary to ', save_path, '\n')

        write.csv(model_summary, file = save_path, row.names = FALSE)
    ''')
    return pd.read_csv(save_directory)

class ApolloEstimator:
    def __init__(self, dataset, output_dir, attributes, covariates):
        from pathlib import Path
        self.dataset = Path(dataset).resolve()
        if not self.dataset.is_file():
            raise FileNotFoundError(self.dataset)
        self.output_dir = str(output_dir)
        self.attributes = attributes
        self.covariates = covariates

    def __call__(self, representation, before_estimate):
        codes = representation.split('_')
        state = [int(c[0]) for c in codes]
        specific = [int(c[1]) for c in codes]
        covariates = [int(c[2:]) for c in codes]
        fixed = create_apollo_fixed(state, specific, covariates, self.attributes, self.covariates)
        estimated = create_apollo_non_fixed(fixed, self.attributes, self.covariates)
        return mnl_interaction(representation, fixed, estimated, self.attributes,
                               self.covariates, 1, self.output_dir,
                               str(self.dataset), before_estimate=before_estimate).iloc[0].to_dict()
