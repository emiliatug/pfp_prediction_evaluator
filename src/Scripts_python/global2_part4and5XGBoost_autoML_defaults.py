import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, MaxAbsScaler, \
    QuantileTransformer, FunctionTransformer

# Suppress the specific warning by matching its message
warnings.filterwarnings(
    "ignore",
    message="To use pickling first train using float32 data to fit the estimator",
    category=UserWarning,
    module="cuml.internals.api_decorators"
)

import cudf as cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor as curfr
# Imports: GroupD: general: system, and GPU related
import os
import sys
import rmm  # RAPIDS Memory Manager

rmm.reinitialize(pool_allocator=False)  # no allocation

from typing import List, Dict

cp.cuda.Device().use()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.utils import generate_logging_file
from flaml import AutoML
from Modules.Global2_fx_for_part4_xgboost import (preprocess_data_for_gpu1, runs_autoML_on_defaults,
                                                  retrain_and_get_stats_from_params, edit_Rank_OutOfTotal_columns,
                                                  make_features_and_labels
    #       generate_hyperparam_configs, cv_with_best_scaler,
    #   read_reweight_and_preprocess_folds,preprocess_data_for_gpu3,
    #   choose_best_param_per_fold_per_model,generate_dict_with_best_chosen_hyper_param_per_model_per_fold
                                                  )


def mainprocess(
        columns_not_to_scale: list,
        # Not using any for any scaling here, just using it to deal with float64 to the float32 conversion
        ordinal_features_not_to_weight: list,
        # Not using any scaler here, but if I wanted too, I could have added the scaler, and then I would have to edit the code to scale the data
        input3: List[str],
        input4: List[str],
        num_var: List[str],
        best_scaler_var: List[str],
        # Not using any scaler here, but if I wanted too, I could have added the scaler, and then I would have to edit the code to scale the data
        var1_weight_status: str,  # that is eigher weighted or unwegihted by classes
        var2_class: str,
        model_list: List[str],
        org_var: List[str],
        input11: List[str],
        input12: List[str],
        expanded_hyperparams: Dict[str, List]
        # not using this here, but in the other XGboost, the parameters are used, so keep it here
) -> None:
    # set the output dir for the logging file
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info(f"Starting Part4: Running AutoML using Train data and validating it on the Val Data")

    RANDOM_STATE = 42
    scaler_dict = {
        "RobustScaler": RobustScaler(),  # Deterministic, no random state needed
        "StandardScaler": StandardScaler(),  # Deterministic, no random state needed,
        "PowerTransformer": PowerTransformer(method="yeo-johnson", standardize=True),  # Deterministic
        "MaxAbsScaler": MaxAbsScaler(),  # Deterministic, no random state needed
        "MinMaxScaler": MinMaxScaler(),  # Scales features to a given range (default [0, 1]),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE),
        # Fixed random state
        "LogScaler": FunctionTransformer(np.log1p, validate=True)
    }

    for model_name in model_list:
        # make **output_dir_files**, where all the files will be stored
        output_dir_files = org_var[2]
        os.makedirs(output_dir_files, exist_ok=True)
        output_dir = org_var[2]
        os.makedirs(output_dir, exist_ok=True)

    # get the variables names
    grid_search_status = org_var[
        3]  # not using this here, but in the other XGboost, the parameters are used, so keep it here
    num_folds = num_var[0]
    number_chosen_bins = num_var[1]
    model_name = model_list[0]
    time_budget = int(org_var[9])
    log_auto_ml_file_name = str(org_var[10])

    #
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
    logger.info("All the variables and loaded")
    logger.info(f"Shape of the **train_df** is {CV0_train_df.shape} and **val_df** {CV0_val_df.shape}")

    print("input is downsampled", CV4_val_df.shape, CV4_train_df.shape)

    # Step1: generate dict_cv_data,: I left the weight parameter in, but in most cases (all except OLS), it made it 'unweighted'. In case I had to go back
    dict_cv_data = {}
    fold_data = [CV0_val_df, CV1_val_df, CV2_val_df, CV3_val_df, CV4_val_df, CV0_train_df, CV1_train_df, CV2_train_df,
                 CV3_train_df, CV4_train_df]
    fold_data_names = ["CV0_val_df_fl32", "CV1_val_df_fl32", "CV2_val_df_fl32", "CV3_val_df_fl32", "CV4_val_df_fl32",
                       "CV0_train_df_fl32", "CV1_train_df_fl32", "CV2_train_df_fl32", "CV3_train_df_fl32",
                       "CV4_train_df_fl32"]
    for each_fold_data, each_fold_data_name in zip(fold_data, fold_data_names):
        each_fold_data = each_fold_data.drop(
            ['Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term', 'Name', 'BinaryScore'], axis=1)
        each_fold_data_tmp = preprocess_data_for_gpu1(each_fold_data, columns_not_to_scale)
        dict_cv_data[each_fold_data_name] = each_fold_data_tmp
    del CV0_val_df, CV1_val_df, CV2_val_df, CV3_val_df, CV4_val_df, CV0_train_df, CV1_train_df, CV2_train_df, CV3_train_df, CV4_train_df

    dict_cv_data_cpu = edit_Rank_OutOfTotal_columns(dict_cv_data, org_var[6], num_folds)

    logger.info(
        "Step1 and 2: the train and val data are made into a dictionary and then float64 is changed to the float 32 and the Rank and OutOfTotal columns are edited")

    print("keys of the **dict_cv_data** for part4 are ", dict_cv_data.keys())
    print("keys of the **dict_cv_data_cpu** for part4 are ", dict_cv_data_cpu.keys())
    print("keys of the **dict_cv_data_cpu** for part4 are ", dict_cv_data_cpu["train"].keys())
    print("keys of the **dict_cv_data_cpu** for part4 are ", dict_cv_data_cpu["val"].keys())

    list_autoML_defaults_models_df, list_autoML_defaults_models_df_with_metrics = [], []
    for fold_idx in range(0, num_folds):
        # The iteration in folds, allows the data to process by fold, the autoML makes fold-wise predictions and they they are evaluted fold-wise by the GXBoost
        train_X_input, train_y_input, val_X_input, val_y_input = make_features_and_labels(dict_cv_data_cpu, org_var[6],
                                                                                          fold_idx, logger)
        if model_name == "xgboost_autoOptuna":
            print("Model name is", model_name)

        elif model_name == "xgboost_autoFlaml":  # used float64
            # print(train_y_input.dtypes)
            print("Model name is", model_name)
            # Follow the pathways for the AutoML
            autoML_defaults_models_per_cv = runs_autoML_on_defaults(time_budget, log_auto_ml_file_name, train_X_input,
                                                                    train_y_input, val_X_input, val_y_input, fold_idx)
            print(
                f"For fold {fold_idx}, this is autoML optuna generated df for train and val for part4 {autoML_defaults_models_per_cv.head()}")

        print(
            f"X_train shape: {train_X_input.shape} y_train shape: {train_y_input.shape} X_val shape: {val_X_input.shape} y_val shape: {val_y_input.shape}")
        if train_X_input.dtypes.equals(val_X_input.dtypes):  # Works for Pandas DataFrames
            print("X_train and X_val have matching data types.")
        else:
            print("Mismatch in X data types! Fixing it below and checking it again")
            val_X_input = val_X_input.astype(train_X_input.dtypes)
            print("X_train and X_val have matching data types.")

        # Check for NaN values: there were issues with the NaN in the tutorial
        print(f"NaN values in X_train: {train_X_input.isna().sum().sum()}")
        print(f"NaN values in y_train: {train_y_input.isna().sum().sum()}")
        print(f"NaN values in X_val: {val_X_input.isna().sum().sum()}")
        print(f"NaN values in y_val: {val_y_input.isna().sum().sum()}")
        logger.info(
            "Step3: The train and val data (both X and y) is regenrated from the dictionary data quality is checked. The AutoMLis run on the train data and the val data on defaults and the parameters are saved in a df.")
        autoML_defaults_models_cv_with_metrics, _ = retrain_and_get_stats_from_params(autoML_defaults_models_per_cv,
                                                                                      train_X_input, train_y_input,
                                                                                      val_X_input, val_y_input,
                                                                                      number_chosen_bins, fold_idx)

        print(autoML_defaults_models_cv_with_metrics.shape)
        print(autoML_defaults_models_cv_with_metrics)
        print(
            f"For fold {fold_idx}, this is autoML generated df for train and val with stats for part4 {autoML_defaults_models_cv_with_metrics.head()}")
        logger.info(
            "Step4: The parameters from the previous step are used to run the XGBoost on the train and val data and the stats are saved")
        list_autoML_defaults_models_df.append(autoML_defaults_models_per_cv)
        list_autoML_defaults_models_df_with_metrics.append(autoML_defaults_models_cv_with_metrics)
    fiveCV_autoML_defaults_models_df = pd.concat(list_autoML_defaults_models_df, axis=0)
    stats_for_model_autoML_defaults = pd.concat(list_autoML_defaults_models_df_with_metrics, axis=0)

    fiveCV_autoML_defaults_models_df.to_pickle(f'{output_dir_files}/fiveCV_autoML_defaults_models_df.pkl', protocol=4)
    stats_for_model_autoML_defaults.to_pickle(
        f"{output_dir_files}/stats_for_model_autoML_defaults_{var2_class}_{var1_weight_status}_df.pkl", protocol=4)
    logger.info("Step5 and Step6: results from different folds are concatenated and saved")
    del train_X_input, train_y_input, val_X_input, val_y_input

    # #########################################################Part 5#####################################################################################################################
    logger.info(
        "Starting Part5: Running XGBoost's top 1 model in each fold on the Train+val Data, and validating it on the Test Data")

    output_dir_files = org_var[4]
    os.makedirs(output_dir_files, exist_ok=True)

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
    logger.info("All the variables and loaded")
    logger.info(f"Shape of the **train_val_df** is {CV0_train_val_df.shape} and **test_df** {CV0_test_df.shape}")

    output_dir = org_var[4]
    os.makedirs(output_dir, exist_ok=True)
    grid_search_status = org_var[
        5]  ##not using this here, but in the other XGboost, the parameters are used, so keep it her
    num_folds = num_var[0]
    number_chosen_bins = num_var[1]

    dict_cv_data = {}
    fold_data = [CV0_test_df, CV1_test_df, CV2_test_df, CV3_test_df, CV4_test_df, CV0_train_val_df, CV1_train_val_df,
                 CV2_train_val_df, CV3_train_val_df, CV4_train_val_df]
    fold_data_names = ["CV0_test_df_fl32", "CV1_test_df_fl32", "CV2_test_df_fl32", "CV3_test_df_fl32",
                       "CV4_test_df_fl32", "CV0_train_val_df_fl32", "CV1_train_val_df_fl32", "CV2_train_val_df_fl32",
                       "CV3_train_val_df_fl32", "CV4_train_val_df_fl32"]
    for each_fold_data, each_fold_data_name in zip(fold_data, fold_data_names):
        each_fold_data = each_fold_data.drop(
            ['Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term', 'Name', 'BinaryScore'], axis=1)
        each_fold_data_tmp = preprocess_data_for_gpu1(each_fold_data, columns_not_to_scale)
        dict_cv_data[each_fold_data_name] = each_fold_data_tmp
    del CV0_test_df, CV1_test_df, CV2_test_df, CV3_test_df, CV4_test_df, CV0_train_val_df, CV1_train_val_df, CV2_train_val_df, CV3_train_val_df, CV4_train_val_df

    dict_cv_data_cpu = edit_Rank_OutOfTotal_columns(dict_cv_data, org_var[7], num_folds)

    logger.info(
        "Step1 and 2: the train_val and test data are made into a dictionary and then float64 is changed to the float 32 and the Rank and OutOfTotal columns are edited")

    print("Keys of the **dict_cv_data** for part5 are ", dict_cv_data.keys())
    print("Keys of the **dict_cv_data_cpu** for part5 are ", dict_cv_data_cpu.keys())
    print("Keys of thee **dict_cv_data_cpu** for part5 are ", dict_cv_data_cpu["train_val"].keys())
    print("Keys of thee **dict_cv_data_cpu** for part5 are ", dict_cv_data_cpu["test"].keys())

    list_best_model_autoML_defaults = []
    for fold_idx in range(0, num_folds):
        train_val_X_input, train_val_y_input, test_X_input, test_y_input = make_features_and_labels(dict_cv_data_cpu,
                                                                                                    org_var[7],
                                                                                                    fold_idx, logger)

        print(
            f"X_train shape: {train_val_X_input.shape} y_train shape: {train_val_y_input.shape} X_val shape: {test_X_input.shape} y_val shape: {test_y_input.shape}")
        if train_val_X_input.dtypes.equals(test_X_input.dtypes):  # Works for Pandas DataFrames
            print("X_train and X_val have matching data types.")
        else:
            print("Mismatch in X data types! Fixing it below and checking it again")
            test_X_input = test_X_input.astype(train_val_X_input.dtypes)
            print("X_train and X_val have matching data types.")

        # Check for NaN values: there were issues with the NaN in the tutorial
        print(f"NaN values in X_train: {train_val_X_input.isna().sum().sum()}")
        print(f"NaN values in y_train: {train_val_y_input.isna().sum().sum()}")
        print(f"NaN values in X_val: {test_X_input.isna().sum().sum()}")
        print(f"NaN values in y_val: {test_y_input.isna().sum().sum()}")
        logger.info(
            "Step3: The train_val and test data (both X and y) is regenerated from the dictionary data quality is checked.")

        eachCV_autoML_defaults_models_df_with_metrics = stats_for_model_autoML_defaults[
            stats_for_model_autoML_defaults["Fold"] == fold_idx]
        eachCV_autoML_defaults_models_df_with_metrics_sorted = eachCV_autoML_defaults_models_df_with_metrics.sort_values(
            by=["Weighted Testing R²"], ascending=False).head(1)
        logger.info(
            "Step4: The df with all the parameters with all the folds (genereated in part4) is filtered to only have the data for the specific fold, and then the best model is chosen according to the Weighted R2")
        best_model_autoML_defaults_df, model_autoML_defaults = retrain_and_get_stats_from_params(
            eachCV_autoML_defaults_models_df_with_metrics_sorted, train_val_X_input, train_val_y_input, test_X_input,
            test_y_input, number_chosen_bins, fold_idx)
        logger.info(
            "Step5: The parameters for a single best model from each fold (from the previous step) are used to run the XGBoost on the train_val and test data and the stats are saved")
        list_best_model_autoML_defaults.append(best_model_autoML_defaults_df)

        with open(f"{org_var[4]}/model_autoML_defaults_fold{fold_idx}_BalB_unweighted.pkl", "wb") as file:
            pickle.dump(model_autoML_defaults, file, protocol=4)
    stats_for_model_autoML_defaults = pd.concat(list_best_model_autoML_defaults, axis=0).reset_index(drop=True)
    stats_for_model_autoML_defaults.to_pickle(
        f"{output_dir_files}/stats_for_model_autoML_defaults_{var2_class}_{var1_weight_status}_df.pkl", protocol=4)
    logger.info("Step6 and Step7: results from different folds are concatenated and saved")


if __name__ == "__main__":
    if len(sys.argv) != 14:
        print(
            "Usage: python global2_part4and5XGBoost.py <Columns_not_to_scale> <Ordinal_features_not_to_weight> <input3_arg List> <input4_arg> <num_folds>, <best_scaler_var>,<var1_weight_status>,<var2_class>,<model_list>,<org_var>, <expanded_hyperparams>")
        sys.exit(1)

    columns_not_to_scale = sys.argv[1]
    ordinal_features_not_to_weight = sys.argv[2]
    input3_arg = sys.argv[3]
    input4_arg = sys.argv[4]
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

    mainprocess(columns_not_to_scale, ordinal_features_not_to_weight, input3, input4, num_var, best_scaler_var
                , var1_weight_status, var2_class, model_list, org_var, input11, input12, expanded_hyperparams)
