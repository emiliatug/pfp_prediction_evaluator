import gc
import json
# Imports: GroupD: general: system, and GPU related
import os
import pickle
import sys

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

rmm.reinitialize(pool_allocator=True, initial_pool_size=18 * 1024 ** 3)  # Use up to 16 GB of GPU memory
from typing import List, Dict

cp.cuda.Device().use()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.utils import generate_logging_file
from cuml.svm import LinearSVR
from cuml.svm import SVR
from cuml.internals.logger import set_level, level_trace
from Modules.Global2_fx_for_linear_models_script import (cv_with_gridserch_for_linear_fx,
                                                         read_reweight_and_preprocess_folds, preprocess_data_for_gpu3,
                                                         preprocess_data_for_gpu3_5,
                                                         choose_best_param_per_fold_per_model,
                                                         generate_dict_with_best_chosen_hyper_param_per_model_per_fold)
from Modules.Global2_fx_for_part4_rf import generate_hyperparam_configs_general, cv_per_model_svr


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
        org_var: List[str],
        input11: List[str],
        input12: List[str],
        expanded_hyperparams: Dict[str, List]
) -> None:
    # set the output dir for the logging file
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info(f"Starting the Global Part2: Best-Parameter Selection: Running {model_list[0]} Model Only")

    for model_name in model_list:
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
        set_level(level_trace)
        model_dict = {
            "svr_kernel_linear": lambda **kwargs: SVR(**kwargs),
            "svr_kernel_rbf": lambda **kwargs: SVR(**kwargs),
            "svr_linear_model": lambda **kwargs: LinearSVR(**kwargs)
        }

        scaler_dict = {
            "MaxAbsScaler": MaxAbsScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
            "PowerTransformer": PowerTransformer(method="yeo-johnson", standardize=True),
            "StandardScaler": StandardScaler(),
            "LogScaler": FunctionTransformer(np.log1p, validate=True)
        }

        # log the locations of all the used files
        for i, file in enumerate(input3):
            logger.info(f"Absolute path of input3[{i}]: {os.path.abspath(file)}")
        for i, file in enumerate(input4):
            logger.info(f"Absolute path of input4[{i}]: {os.path.abspath(file)}")

        CV0_train_df = pd.read_pickle(input3[0])
        CV1_train_df = pd.read_pickle(input3[1])
        CV2_train_df = pd.read_pickle(input3[2])
        CV3_train_df = pd.read_pickle(input3[3])
        CV4_train_df = pd.read_pickle(input3[4])

        CV0_val_df = pd.read_pickle(input4[0])
        CV1_val_df = pd.read_pickle(input4[1])
        CV2_val_df = pd.read_pickle(input4[2])
        CV3_val_df = pd.read_pickle(input4[3])
        CV4_val_df = pd.read_pickle(input4[4])

        logger.info(f"Shape of the **train_df** is {CV0_train_df.shape} and **val_df** {CV0_val_df.shape}")
        logger.info("All the variables and loaded")
        logger.info("The approximate shape of train data is 400k. Is above shape 400K?")

        # Step1: generate dict_cv_data,best_scaler_per_model_df: I left the weight parameter in, but in most cases (all except OLS), it made it 'unweighted'. In case I had to go back
        dict_folds_with_weights = read_reweight_and_preprocess_folds(
            # Validation Data (List of 5 DataFrames, one per fold)
            [
                CV0_val_df,  # Validation DataFrame for Fold 0
                CV1_val_df,  # Validation DataFrame for Fold 1
                CV2_val_df,  # Validation DataFrame for Fold 2
                CV3_val_df,  # Validation DataFrame for Fold 3
                CV4_val_df  # Validation DataFrame for Fold 4
            ],
            # Training Data (List of 5 DataFrames, one per fold)
            [
                CV0_train_df,  # Training DataFrame for Fold 0
                CV1_train_df,  # Training DataFrame for Fold 1
                CV2_train_df,  # Training DataFrame for Fold 2
                CV3_train_df,  # Training DataFrame for Fold 3
                CV4_train_df  # Training DataFrame for Fold 4
            ],
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
            "Step1 (for Best-Parameter Selection): fx 'read_reweight_and_preprocess_folds' is done. A dict is gen, it can be either weighted or not depending on the arg12; not yet scaled, or in float32")
        # Fx call 2: dict_cv_data_cpu_, a  dict (that is either weighted or not, depending on the arg12; not yet scaled, or in float32)
        dict_cv_data_cpu = preprocess_data_for_gpu3(
            best_scaler_var[0],  # Best scaler variable used for preprocessing
            model_name,  # Model name (e.g., "lasso", "ridge")
            num_folds,  # Number of cross-validation folds
            dict_folds_with_weights,  # Dictionary of training/validation folds with sample weights
            columns_not_to_scale,  # List of columns that should not be scaled
            model_dict,  # Dictionary containing model-related parameters
            scaler_dict,  # Scaling methods (e.g., StandardScaler, MinMaxScaler)
            var1_weight_status
            # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
        )
        if best_scaler_var[1] != "None":
            print("should not be here")
            dict_cv_data_cpu1 = preprocess_data_for_gpu3_5(
                logger,
                best_scaler_var[1],  # Best scaler variable used for preprocessing
                model_name,  # Model name (e.g., "lasso", "ridge")
                num_folds,  # Number of cross-validation folds
                dict_cv_data_cpu,  # Dictionary of training/validation folds with sample weights
                columns_not_to_scale,  # List of columns that should not be scaled
                model_dict,  # Dictionary containing model-related parameters
                scaler_dict,  # Scaling methods (e.g., StandardScaler, MinMaxScaler)
                var1_weight_status
                # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
            )

            dict_cv_data_cpu = dict_cv_data_cpu1

        logger.info(
            "Step2 (for Best-Parameter Selection): fx 'preprocess_data_for_gpu3' is done, and the 'dict_cv_data_cpu' is created, but not written out to the hard drive")
        # Step3: This is a main step, that takes each model with a number of bins, and the df_val_with_weights (for the weighted mse) and performs a grid search to generate df with
        organized_hyperparams_dict = generate_hyperparam_configs_general(logger, expanded_hyperparams, model_dict,
                                                                         model_name)
        with open(f'{output_dir_files}/organized_hyperparams_dict.pkl', 'wb') as f:
            pickle.dump(organized_hyperparams_dict, f)
        logger.info("Step3: Organized_hyperparams_dict is created, and is saved")

        RANDOM_STATE = 42
        final_list = []
        for model_name in model_list:
            single_model_all_trained_parameters_df, full_dict = cv_per_model_svr(
                model_dict,
                RANDOM_STATE,
                scaler_dict,
                best_scaler_var,
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
                model_name,  # Current model being processed (e.g., "lasso", "ridge")
                dict_cv_data_cpu,  # Preprocessed CV data stored in CPU memory
                columns_not_to_scale=columns_not_to_scale,  # Features that should not be scaled
                num_folds=num_folds  # Number of cross-validation folds
            )
            final_list.append(single_model_all_trained_parameters_df)
            #     print_avail_GPU_memory()

        #     #Step4B: Generating a big df and saving it
        stats_model_tmp_df = pd.concat(final_list, axis=0).reset_index(drop=True)
        stats_model_tmp_df.to_pickle(
            f'{output_dir_files}/stats_for_model_{model_name}_{var2_class}_{var1_weight_status}_df.pkl', protocol=4)
        logger.info("Step4 is finished and the df comparisons is saved. The trained models are not saved")
        logger.info(
            f"**Global2_part4** fx is done for model: {model_name} with the Downsampled & Weighted Params: {var1_weight_status} and {var2_class}")

        # DELETEDELETEDELETE
        pd.set_option('display.max_columns', None)
        print("training data params", stats_model_tmp_df.shape)
        print(stats_model_tmp_df)

        #     #Step5: Selecting a best param_key by the a)weighted R2 and weighted MSE and saving it b)choosing default if 2+ best params c)choosing top if in best 2+ no default
        dict_best_param_key_per_fold = choose_best_param_per_fold_per_model(
            logger,
            model_name,
            stats_model_tmp_df,
            "Fold",
            "Weighted Validation R^2",
            "Weighted Validation MSE",
            "Param"
        )

        with open(f"{output_dir_files}/dict_best_param_key_per_fold.pkl", 'wb') as handle:
            pickle.dump(dict_best_param_key_per_fold, handle, protocol=4)
        print("prefatso1")
        print("dict_best_param_key_per_fold", dict_best_param_key_per_fold)
        print()
        print("organized_hyperparams_dict", organized_hyperparams_dict)
        dict_best_parameters_values_per_fold = generate_dict_with_best_chosen_hyper_param_per_model_per_fold(
            dict_best_param_key_per_fold, full_dict)
        print("fatso1")
        with open(f'{output_dir_files}/dict_best_parameters_values_per_fold.pkl', 'wb') as handle:
            pickle.dump(dict_best_parameters_values_per_fold, handle, protocol=4)
        logger.info(
            f"Part4 for {model_list[0]} Model is done, and each_fold:best_param_keys, saved as dict **dict_best_param_key_per_fold** on the train and val data")

    del CV0_val_df, CV1_val_df, CV2_val_df, CV3_val_df, CV4_val_df, CV0_train_df, CV1_train_df, CV2_train_df, CV3_train_df, CV4_train_df, dict_cv_data_cpu
    gc.collect()
    logger.info("")
    if org_var[-1] == "True":
        print("Initial_NotExpanded has been processed")
        quit()

    #########################################################Part 5#####################################################################################################################
    logger.info("")
    logger.info(f"Starting Best-Parameter Usage training using train_val and test data for {model_list[0]}")
    for model in model_list:
        # Step6: Reading the **dict_best_param_key_per_fold**, a dict that contains the best hyper params_keys as selected by the train/val data, creted by the previous part
        # and **organized_hyperparams_dict**, a dict that contains all the hyper param combs in order to generate a shorter dict that only contains best hyper param with the values.  
        # Step6a: Reads the 2 dicts 
        with open(f"{output_dir_files}/dict_best_param_key_per_fold.pkl", 'rb') as handle:
            dict_best_param_key_per_fold = pickle.load(handle)
        with open(f"{output_dir_files}/organized_hyperparams_dict.pkl", 'rb') as handle:
            organized_hyperparams_dict = pickle.load(handle)
        logger.info("Step6 (for the Best-Parameter Usage) is done")

        output_dir_files = org_var[6]
        os.makedirs(output_dir_files, exist_ok=True)

        # Step8 Reading the train_val and test data
        for i, file in enumerate(input11):
            logger.info(f"Absolute path of input11[{i}]: {os.path.abspath(file)}")
        for i, file in enumerate(input12):
            logger.info(f"Absolute path of input12[{i}]: {os.path.abspath(file)}")

        CV0_train_val_df = pd.read_pickle(input11[0])
        CV1_train_val_df = pd.read_pickle(input11[1])
        CV2_train_val_df = pd.read_pickle(input11[2])
        CV3_train_val_df = pd.read_pickle(input11[3])
        CV4_train_val_df = pd.read_pickle(input11[4])

        CV0_test_df = pd.read_pickle(input12[0])
        CV1_test_df = pd.read_pickle(input12[1])
        CV2_test_df = pd.read_pickle(input12[2])
        CV3_test_df = pd.read_pickle(input12[3])
        CV4_test_df = pd.read_pickle(input12[4])

        logger.info(f"Shape of the **train_val_df** is {CV0_train_val_df.shape} and **test_df** {CV0_test_df.shape}")
        logger.info("Step8 (for the Best-Parameter Usage) is done")

        output_dir = org_var[4]
        os.makedirs(output_dir, exist_ok=True)
        grid_search_status = org_var[5]
        num_folds = num_var[0]
        number_chosen_bins = num_var[1]

        print(model_dict)
        print(scaler_dict)

        # step9: generate dict_cv_data,best_scaler_per_model_df: I left the weight parameter in, but in most cases (all except OLS), it made it 'unweighted'
        dict_folds_with_weights = read_reweight_and_preprocess_folds([
            CV0_test_df,
            CV1_test_df,
            CV2_test_df,
            CV3_test_df,
            CV4_test_df],

            [CV0_train_val_df,
             CV1_train_val_df,
             CV2_train_val_df,
             CV3_train_val_df,
             CV4_train_val_df],
            num_folds,
            number_chosen_bins,
            ordinal_features_not_to_weight,
            columns_not_to_scale,
            model_list,
            var1_weight_status)
        logger.info("Step9 (for the Best-Parameter Usage) is done")

        # Step10: scale a inputs data using a Log scaler
        dict_cv_data_cpu = preprocess_data_for_gpu3(
            best_scaler_var[0],  # Best scaler variable used for preprocessing
            model_name,  # Model name (e.g., "lasso", "ridge")
            num_folds,  # Number of cross-validation folds
            dict_folds_with_weights,  # Dictionary of training/validation folds with sample weights
            columns_not_to_scale,  # List of columns that should not be scaled
            model_dict,  # Dictionary containing model-related parameters
            scaler_dict,  # Scaling methods (e.g., StandardScaler, MinMaxScaler)
            var1_weight_status
            # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
        )
        # Step11: scale a inputs data using a Standard scaler
        if best_scaler_var[1] != "None":
            print("should not be here")
            dict_cv_data_cpu1 = preprocess_data_for_gpu3_5(
                logger,
                best_scaler_var[1],  # Best scaler variable used for preprocessing
                model_name,  # Model name (e.g., "lasso", "ridge")
                num_folds,  # Number of cross-validation folds
                dict_cv_data_cpu,  # Dictionary of training/validation folds with sample weights
                columns_not_to_scale,  # List of columns that should not be scaled
                model_dict,  # Dictionary containing model-related parameters
                scaler_dict,  # Scaling methods (e.g., StandardScaler, MinMaxScaler)
                var1_weight_status
                # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
            )
            dict_cv_data_cpu = dict_cv_data_cpu1
        logger.info("Step10 and Step11 (for the Best-Parameter Usage) is done")

    RANDOM_STATE = 42
    final_list = []
    for model_name in model_list:
        single_model_all_trained_parameters_df, full_dict = cv_per_model_svr(
            model_dict,
            RANDOM_STATE,
            scaler_dict,
            best_scaler_var,
            logger,
            org_var[5],
            # Determines if part4 or part5 is happening. Part4 is grid search and part5 is application of the best param keys chosen by the grid search
            number_chosen_bins,  # Number of bins used for reweighting
            # weights_for_val_df,  # Validation dataset with weights applied
            grid_search_status,  # Status of grid search (Grid-Search and NoGrid-Search)
            var2_class,  # Classification of initial Data (A, or B)
            dict_best_parameters_values_per_fold,  # Dictionary of organized hyperparameters per model
            var1_weight_status,
            # Whether to the raw dataset should re reclassified (at the very begining): This option only exists in OLS. For the rest of them, its 'unweighted'
            output_dir_files,  # Directory where output files are stored
            model_name,  # Current model being processed (e.g., "lasso", "ridge")
            dict_cv_data_cpu,  # Preprocessed CV data stored in CPU memory
            columns_not_to_scale=columns_not_to_scale,  # Features that should not be scaled
            num_folds=num_folds  # Number of cross-validation folds
        )

    final_list.append(single_model_all_trained_parameters_df)

    #     #Step4B: Generating a big df and saving it
    stats_model_tmp_df = pd.concat(final_list, axis=0).reset_index(drop=True)
    # saving the final stats for the part5
    stats_model_tmp_df.to_pickle(
        f'{output_dir_files}/stats_for_model_{model_name}_{var2_class}_{var1_weight_status}_df.pkl', protocol=4)
    logger.info(
        "Part5 (Train_val and Test): Step4 is finished and the df comparisons is saved. Trained models are saved in part5, but in a diff step")
    logger.info(
        f"Part5 (Train_val and Test) is done for model: {model_name} with the Downsampled & Weighted Params: {var1_weight_status} and {var2_class}")
    print("test data params", stats_model_tmp_df.shape)
    print(stats_model_tmp_df)


if __name__ == "__main__":
    if len(sys.argv) != 14:
        print(
            "Usage: python Part2_5.py <Columns_not_to_scale> <Ordinal_features_not_to_weight> <input3_arg List> <input4_arg> <num_folds>, <best_scaler_var>,<var1_weight_status>,<var2_class>,<model_list>,<org_var>,<expanded_hyperparams>")
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
    input11_arg = sys.argv[11]
    input12_arg = sys.argv[12]
    expanded_hyperparams = json.loads(sys.argv[13])

    num_var = list(map(int, sys.argv[5].split(',')))
    columns_not_to_scale = [col.strip() for col in columns_not_to_scale.split(",")]
    ordinal_features_not_to_weight = [col.strip() for col in ordinal_features_not_to_weight.split(",")]
    input3 = [col.strip() for col in input3_arg.split(",")]  # Now each file has the full directory path
    input4 = [col.strip() for col in input4_arg.split(",")]
    best_scaler_var = [col.strip() for col in best_scaler_var.split(",")]
    model_list = [col.strip() for col in model_list.split(",")]
    org_var = [col.strip() for col in org_var.split(",")]
    input11 = [col.strip() for col in input11_arg.split(",")]  # Now each file has the full directory path
    input12 = [col.strip() for col in input12_arg.split(",")]
    # list_input_alphas=json.loads(sys.argv[13])

    mainprocess(columns_not_to_scale, ordinal_features_not_to_weight, input3, input4, num_var, best_scaler_var
                , var1_weight_status, var2_class, model_list, org_var, input11, input12, expanded_hyperparams)
