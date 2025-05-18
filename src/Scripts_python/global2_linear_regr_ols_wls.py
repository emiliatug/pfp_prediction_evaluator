import numpy as np
# Imports: GroupD: general: system, and GPU related
import os
import sys
# rmm.reinitialize(pool_allocator=True, initial_pool_size=18 * 1024 ** 3)  # Use up to 16 GB of GPU memory
from typing import List

# Imports: GroupC: specific library: cuml
import cudf as cudf
import cupy as cp
import numpy as np
import pandas as pd
import rmm  # RAPIDS Memory Manager
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
# Imports: GroupB: specific library: sklearn
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, MaxAbsScaler, \
    QuantileTransformer, FunctionTransformer

cp.cuda.Device().use()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.utils import generate_logging_file
from Modules.Global2_fx_for_linear_models_script import (cv_with_gridserch_for_linear_fx,
                                                         read_reweight_and_preprocess_folds, preprocess_data_for_gpu3
                                                         )


def mainprocess(
        columns_not_to_scale: list,
        ordinal_features_not_to_weight: list,
        input3: List[str],
        input4: List[str],
        num_var: List[str],
        best_scaler_var: List[str],
        var1_weight_status: str,  # that is eigher weighted or unwegihted by classes
        var2_class: str,
        model_list: List[str],
        org_var: List[str]
) -> None:
    # set the output dir for the logging file
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info(f"Starting the Global Part2: Part5: Running {model_list[0]} Model Only")

    for model in model_list:
        # make **output_dir_files**, where all the files will be stored
        output_dir_files = org_var[2]
        os.makedirs(output_dir_files, exist_ok=True)
        output_dir = org_var[2]
        os.makedirs(output_dir, exist_ok=True)

        # get the variables names
        grid_search_status = org_var[3]
        num_folds = num_var[0]
        number_chosen_bins = num_var[1]

        # get the needed fx
        model_dict = {
            "ols": LinearRegression(copy_X=True),
            "mb_SGD": cumlMBSGDRegressor(penalty="none"),
            "lasso": cumlMBSGDRegressor(penalty="l1"),
            "ridge": cumlMBSGDRegressor(penalty="l2"),
            "elasticnet": cumlMBSGDRegressor(penalty="elasticnet")
        }

        scalers = {
            "MaxAbsScaler": MaxAbsScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
            "PowerTransformer": PowerTransformer(method="yeo-johnson", standardize=True),
            "StandardScaler": StandardScaler(),
            "LogScaler": FunctionTransformer(np.log1p, validate=True),
            "IdentityScaler": FunctionTransformer()
        }

        # log the locations of all the used files
        for i, file in enumerate(input3):
            logger.info(f"Absolute path of input3[{i}]: {os.path.abspath(file)}")
        for i, file in enumerate(input4):
            logger.info(f"Absolute path of input4[{i}]: {os.path.abspath(file)}")

        CV0_train_val_df = pd.read_pickle(input3[0])
        CV1_train_val_df = pd.read_pickle(input3[1])
        CV2_train_val_df = pd.read_pickle(input3[2])
        CV3_train_val_df = pd.read_pickle(input3[3])
        CV4_train_val_df = pd.read_pickle(input3[4])

        CV0_test_df = pd.read_pickle(input4[0])
        CV1_test_df = pd.read_pickle(input4[1])
        CV2_test_df = pd.read_pickle(input4[2])
        CV3_test_df = pd.read_pickle(input4[3])
        CV4_test_df = pd.read_pickle(input4[4])

        logger.info("All the variables and loaded")

        print("shape of the CV0_train_val_df", CV0_train_val_df.shape)
        print("shape of the CV0_test_df", CV0_test_df.shape)
        # Fx call 1: generate dict_cv_data,best_scaler_per_model_df
        dict_folds_with_weights = read_reweight_and_preprocess_folds([
            CV0_test_df,  # Test DataFrame for Fold 0
            CV1_test_df,  # Test DataFrame for Fold 1
            CV2_test_df,  # Test DataFrame for Fold 2
            CV3_test_df,  # Test DataFrame for Fold 3
            CV4_test_df],  # Test DataFrame for Fold 4

            [CV0_train_val_df,  # Training_Val DataFrame for Fold 0
             CV1_train_val_df,  # Training_Val DataFrame for Fold 1
             CV2_train_val_df,  # Training_Val DataFrame for Fold 2
             CV3_train_val_df,  # Training_Val DataFrame for Fold 3
             CV4_train_val_df],  # Training_Val DataFrame for Fold 4
            num_folds,  # Number of cross-validation folds
            number_chosen_bins,  # Number of bins used for reweighting
            ordinal_features_not_to_weight,  # Ordinal features that should not be reweighted
            columns_not_to_scale,  # Features that should not be scaled
            model_list,
            # List of models being used (e.g., "lasso", "ridge", etc. at a time). Here only a single model is used
            var1_weight_status
            # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
        )

        logger.info(
            "Step1: fx 'read_reweight_and_preprocess_folds' is done. A dict is gen, it can be either weighted or not depending on the arg12; not yet scaled, or in float32")

        # Fx call 2: dict_cv_data_cpu_, a  dict (that is either weighted or not, depending on the arg12; not yet scaled, or in float32)
        dict_cv_data_cpu = preprocess_data_for_gpu3(
            best_scaler_var[0],  # Best scaler variable used for preprocessing
            model,  # Model name (e.g., "lasso", "ridge")
            num_folds,  # Number of cross-validation folds
            dict_folds_with_weights,  # Dictionary of training/validation folds with sample weights
            columns_not_to_scale,  # List of columns that should not be scaled
            model_dict,  # Dictionary containing model-related parameters
            scalers,  # Scaling methods (e.g., StandardScaler, MinMaxScaler)
            var1_weight_status
            # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
        )

        logger.info(
            "Step2: fx 'preprocess_data_for_gpu3' is done, and the 'dict_cv_data_cpu' is created, but not written out to the hard drive")
        logger.info("")
        logger.info("*" * 25)
        logger.info("Starting cross-validation with best scalers, solvers, and hyperparameters...")
        logger.info("*" * 25)
        # Step3: Generate grid search parameters: is taken for the step 2.5: the generation only happes in the part4: in part5, the **organized_hyperparams_dict** is not generated
        # organized_hyperparams_dict=generate_hyperparam_configs(logger)
        organized_hyperparams_dict = {}  # I have to send a dict input to the **cv_with_gridserch_for_linear_fx**, so I made an emphty dict

        # Step4: This is a main step, that takes each model with a number of bins, and the df_val_with_weights (for the weighted mse) and performs a grid search to generate df with
        # weighted mse/weighted r2/mse/r2 and param_key names. The individuals models are not saved here.

        # Step4A: Generating dfa list of the df
        final_list = []
        for model in model_list:
            single_model_all_trained_parameters_df = cv_with_gridserch_for_linear_fx(
                logger,
                org_var[4],
                # Determines if part4 or part5 is happening. Part4 is grid search and part5 is application of the best param keys chosen by the grid search
                number_chosen_bins,  # Number of bins used for reweighting
                # weights_for_val_df,  # Validation dataset with weights applied
                grid_search_status,  # Status of grid search (Grid-Search and NoGrid-Search)
                var2_class,  # Classification of initial Data (A, or B)
                organized_hyperparams_dict,  # Dictionary of organized hyperparameters per model
                var1_weight_status,
                # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
                output_dir_files,  # Directory where output files are stored
                scalers,  # Dictionary of scaler objects (e.g., StandardScaler, MinMaxScaler)
                model,  # Current model being processed (e.g., "lasso", "ridge")
                dict_cv_data_cpu,  # Preprocessed CV data stored in CPU memory
                best_scaler_var[0],  # The best scaler chosen for preprocessing
                columns_not_to_scale=columns_not_to_scale,  # Features that should not be scaled
                num_folds=num_folds  # Number of cross-validation folds
            )
        final_list.append(single_model_all_trained_parameters_df)
        #     print_avail_GPU_memory()

        # Step4B: Generating a big df and saving it
        stats_model_tmp_df = pd.concat(final_list, axis=0).reset_index(drop=True)
        stats_model_tmp_df.to_pickle(
            f'{output_dir_files}/stats_for_model_{model}_{var2_class}_{var1_weight_status}_df.pkl', protocol=4)
        logger.info(f"The model {model} is compelelte and trained models are and mse df are saved")


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print(
            "Usage: python Part2_5.py <Columns_not_to_scale> <Ordinal_features_not_to_weight> <input3_arg List> <input4_arg> <num_folds>, <best_scaler_var>,<var1_weight_status>,<var2_class>,<model_list>,<org_var>")
        sys.exit(1)

    columns_not_to_scale = sys.argv[1]
    ordinal_features_not_to_weight = sys.argv[2]
    input3_arg = sys.argv[3]
    input4_arg = sys.argv[4]
    # num_folds = int(sys.argv[5])
    best_scaler_var = sys.argv[6]  # dir and file name, excluding for the model since the models are now run separatly
    var1_weight_status = sys.argv[7]
    var2_class = sys.argv[8]
    model_list = sys.argv[9]
    org_var = sys.argv[10]

    num_var = list(map(int, sys.argv[5].split(',')))
    columns_not_to_scale = [col.strip() for col in columns_not_to_scale.split(",")]
    ordinal_features_not_to_weight = [col.strip() for col in ordinal_features_not_to_weight.split(",")]
    input3 = [col.strip() for col in input3_arg.split(",")]  # Now each file has the full directory path
    input4 = [col.strip() for col in input4_arg.split(",")]
    best_scaler_var = [col.strip() for col in best_scaler_var.split(",")]
    model_list = [col.strip() for col in model_list.split(",")]
    org_var = [col.strip() for col in org_var.split(",")]

    mainprocess(columns_not_to_scale, ordinal_features_not_to_weight, input3, input4, num_var, best_scaler_var
                , var1_weight_status, var2_class, model_list, org_var)
