import copy
import logging
import time
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import List

# Imports: GroupC: specific library: cuml
import cudf as cudf
import cupy as cp
import joblib
import statsmodels.api as sm
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
# Imports: GroupB: specific library: sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, MaxAbsScaler, \
    QuantileTransformer, FunctionTransformer

# Imports: GroupD: general: system, and GPU related
# sys.path.append("../part1_data_generation_fv_02012025")

cp.cuda.Device().use()
import warnings

warnings.filterwarnings("ignore",
                        message="pandas.DataFrame with sparse columns found")  # this just silences a SPECIFIC warnings

# Now import other libraries
import pandas as pd
import numpy as np


# Imports: GroupE: modules import
# from fx_for_python3_5 import splitting_df_to_test_and_train
# Directories set up
# input_dir = "./ML_input_data"
# output_dir="./IntermediateData/Linear_Regression"
# os.makedirs(output_dir, exist_ok=True)

# # pd.set_option('future.no_silent_downcasting', True)
# # pd.set_option('display.max_rows', None)
# # pd.set_option('display.max_columns', None)


##############################################################3#THOSE ARE THE FX FOR THE PART2_3 and 2.4 ###########################################
def calculate_weights(train_folds_df, test_folds_df, number_chosen_bins):
    """This function calculates bin-based statistics for LinScore.
       It returns:
       - Bin Range
       - Count of LinScore values per bin
       - 1 / Count (Inverse Frequency)
       - Normalized Weights ((1/Count) / Total Sum)
    """

    # Remove duplicates to avoid duplicate counts

    train_folds_df = train_folds_df.drop_duplicates()
    original_train_data = train_folds_df.copy()
    y_labels_list = train_folds_df["LinScore"]

    # Define bins from 0 to 1 (101 bins) and ensure last bin edge is slightly increased
    bins = np.linspace(0, 1, number_chosen_bins)
    bins[-1] += 1e-6  # Ensure values exactly at 1.0 are included

    # Compute bin indices
    print("Type of y_labels_list:", type(y_labels_list))
    print("First few y_labels_list values:", y_labels_list[:5])

    print("Type of bins:", type(bins))
    print("First few bins values:", bins[:5])
    bin_indices = np.digitize(y_labels_list, bins, right=False) - 1

    # Ensure values larger than the last bin edge are placed in the last bin
    bin_indices[bin_indices >= len(bins) - 1] = len(bins) - 2  # Assign large values to last valid bin

    # Step 1: Compute Bin Counts
    bin_counts = defaultdict(int)
    for bin_idx in bin_indices:
        bin_counts[bin_idx] += 1

    # Step 2: Compute 1 / Count (Inverse Frequency)
    inverse_counts = {bin_idx: 1 / (count + 1e-6) for bin_idx, count in bin_counts.items()}

    # Step 3: Normalize Weights
    total_sum = sum(inverse_counts.values())
    normalized_weights = {bin_idx: inverse_counts[bin_idx] / total_sum for bin_idx in inverse_counts}

    # Step 4: Create DataFrame of Weights and Bin Ranges
    list_of_labels, list_of_bins, list_of_counts, list_of_inv_counts, list_of_weights = [], [], [], [], []
    for label, bin_idx in zip(y_labels_list, bin_indices):
        bin_range = f"[{bins[bin_idx]:.2f}, {bins[bin_idx + 1]:.2f})"
        list_of_labels.append(label)
        list_of_bins.append(bin_range)
        list_of_counts.append(bin_counts[bin_idx])
        list_of_inv_counts.append(inverse_counts[bin_idx])
        list_of_weights.append(normalized_weights[bin_idx])

    # Create final DataFrame with requested columns
    df_with_weights_train = pd.DataFrame({
        "LinScore": list_of_labels,
        "BinRange": list_of_bins,
        "Count": list_of_counts,
        "InverseCount": list_of_inv_counts,
        "Weight": list_of_weights
    })
    df_with_weights_train_selc_columns_lin_score_weight = df_with_weights_train[
        ["LinScore", "Weight"]].drop_duplicates()
    # Merge weights into the original dataset
    original_train_data_with_weights_column_added = pd.merge(original_train_data,
                                                             df_with_weights_train_selc_columns_lin_score_weight,
                                                             on="LinScore", how="inner")

    return original_train_data_with_weights_column_added


def calculate_weights_mse(test_folds_df, number_chosen_bins):
    """This function calculates bin-based statistics for LinScore specifically for the 2.5 part (test, not ).
       It returns:
       - Bin Range
       - Count of LinScore values per bin
       - 1 / Count (Inverse Frequency)
       - Normalized Weights ((1/Count) / Total Sum)
    """
    # X = test_folds_df.copy, ()
    y_labels_list = test_folds_df.to_pandas()
    # Define bins from 0 to 1 (101 bins) and ensure last bin edge is slightly increased
    bins = np.linspace(0, 1, number_chosen_bins)
    bins[-1] += 1e-6  # Ensure values exactly at 1.0 are included
    # Compute bin indices
    print("Type of y_labels_list:", type(y_labels_list))
    print("First few y_labels_list values:", y_labels_list[:5])

    print("Type of bins:", type(bins))
    print("First few bins values:", bins[:5])
    bin_indices = np.digitize(y_labels_list, bins, right=False) - 1
    # Ensure values larger than the last bin edge are placed in the last bin
    bin_indices[bin_indices >= len(bins) - 1] = len(bins) - 2  # Assign large values to last valid bin
    # Step 1: Compute Bin Counts
    bin_counts = defaultdict(int)
    for bin_idx in bin_indices:
        bin_counts[bin_idx] += 1
    # Step 2: Compute 1 / Count (Inverse Frequency)
    inverse_counts = {bin_idx: 1 / (count + 1e-6) for bin_idx, count in bin_counts.items()}
    # Step 3: Normalize Weights
    total_sum = sum(inverse_counts.values())
    normalized_weights = {bin_idx: inverse_counts[bin_idx] / total_sum for bin_idx in inverse_counts}
    # Step 4: Create DataFrame of Weights and Bin Ranges
    list_of_labels, list_of_bins, list_of_counts, list_of_inv_counts, list_of_weights = [], [], [], [], []
    for label, bin_idx in zip(y_labels_list, bin_indices):
        bin_range = f"[{bins[bin_idx]:.2f}, {bins[bin_idx + 1]:.2f})"
        list_of_labels.append(label)
        list_of_bins.append(bin_range)
        list_of_counts.append(bin_counts[bin_idx])
        list_of_inv_counts.append(inverse_counts[bin_idx])
        list_of_weights.append(normalized_weights[bin_idx])
    # Create final DataFrame with requested columns
    df_with_weights_mse = pd.DataFrame({
        "LinScore": list_of_labels,
        "BinRange": list_of_bins,
        "Count": list_of_counts,
        "InverseCount": list_of_inv_counts,
        "Weight": list_of_weights
    })

    return df_with_weights_mse


def generate_random_subset_for_scaler_exclusion(input1_cudf, input2_cudf, frac, random_seed):
    indices = input1_cudf.sample(frac=frac, random_state=random_seed).index
    input1_cudf = input1_cudf.loc[indices]
    input2_cudf = input2_cudf.loc[indices]
    return input1_cudf, input2_cudf


def generate_stats_df_of_best_scaler(
        model_name: str,
        input_train_X: pd.DataFrame,
        X_weighted: pd.DataFrame,
        input_train_y: pd.Series,
        y_weighted: pd.Series,
        columns_not_to_scale: List[str],
        var1_weight_status: str,
        var2_class: str,
        scaler_list: List[str]
) -> pd.DataFrame:
    """
    Tests various scalers on training data and evaluates model performance for each.
    """
    random_state = 42
    model_dict = {
        "ols": LinearRegression(copy_X=True),
        "mb_SGD": cumlMBSGDRegressor(penalty="none"),
        "lasso": cumlMBSGDRegressor(penalty="l1"),
        "ridge": cumlMBSGDRegressor(penalty="l2"),
        "elasticnet": cumlMBSGDRegressor(penalty="elasticnet")
    }

    scaler_dict = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "PowerTransformer": PowerTransformer(method="yeo-johnson", standardize=True),
        "MaxAbsScaler": MaxAbsScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=random_state),
        "LogScaler": FunctionTransformer(np.log1p, validate=True),
        "IdentityScaler": FunctionTransformer()
    }

    feature_columns = [col for col in input_train_X.columns if col not in columns_not_to_scale]
    results = []
    failed_scaler = []

    for scaler in scaler_list:
        try:
            if var1_weight_status == "weighted":
                column_transformer = ColumnTransformer([
                    (scaler, scaler_dict[scaler], feature_columns)
                ], remainder="passthrough")
                scaled_array = column_transformer.fit_transform(X_weighted)
                scaled_X = pd.DataFrame(scaled_array, columns=X_weighted.columns)
                y = y_weighted
            else:
                column_transformer = ColumnTransformer([
                    (scaler, scaler_dict[scaler], feature_columns)
                ], remainder="passthrough")
                scaled_array = column_transformer.fit_transform(input_train_X)
                scaled_X = pd.DataFrame(scaled_array, columns=input_train_X.columns)
                y = input_train_y

            cudf_X = cudf.DataFrame.from_pandas(scaled_X)
            cudf_y = cudf.Series(y)

            model = model_dict[model_name]
            model.fit(cudf_X, cudf_y)
            predictions = model.predict(cudf_X)

            if predictions.isnull().sum() > 0:
                mse, r2 = float("nan"), float("nan")
                failed_scaler.append([scaler, model_name])
            else:
                mse = mean_squared_error(cudf_y.to_pandas(), predictions.to_pandas())
                r2 = r2_score(cudf_y.to_pandas(), predictions.to_pandas())

            results.append({
                "Model Name": model_name,
                "Scaler": scaler,
                "Validation MSE": mse,
                "Validation R²": r2,
                "NA Count": predictions.isnull().sum()
            })

        except Exception as e:
            logging.error(f"Error for Scaler: {scaler}, Model: {model_name}: {str(e)}")
            failed_scaler.append([scaler, model_name])
            results.append({
                "Model Name": model_name,
                "Scaler": scaler,
                "Validation MSE": float("nan"),
                "Validation R²": float("nan"),
                "NA Count": float("nan")
            })

    return pd.DataFrame(results)


def preprocess_train_and_val_data_for_scalers(input_train_df,
                                              ordinal_features_not_to_weight):  # original code, I forgot to add the weights to the labels
    """This fx takes 2  input df ('input_train_df' a normal train_df, just with the weight parameter added, and 'input_val_df' and after
    processing them all the same way, returns 6 df (x and y)"""

    input_df_W = input_train_df.drop(["Name", 'Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term'], axis=1)
    input_train_df = input_train_df.drop(["Name", 'Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term'], axis=1)
    features_labels = [x for x in list(input_df_W.columns) if x not in ordinal_features_not_to_weight]
    weights_col = "Weight"
    input_df_W[features_labels] = input_df_W[features_labels].astype(float)
    features_matrix = input_train_df[features_labels].values  # Shape: (n_samples, n_features)
    weights_vector = np.sqrt(input_df_W[weights_col].values)  # Shape: (n_samples,)

    # Reshape the weights vector to broadcast properly
    weights_matrix = weights_vector[:, np.newaxis]  # Shape: (n_samples, 1)
    # Perform element-wise multiplication
    weighted_features = features_matrix.astype(float) * weights_matrix.astype(float)  # Shape: (n_samples, n_features)

    # Assign the result back to the DataFrame
    input_df_W[features_labels] = weighted_features

    # train's processing
    input_train_X = input_train_df.drop(["LinScore", "BinaryScore", "Weight"], axis=1)
    input_train_y = input_train_df["LinScore"]
    input_train_X['Out_of_Total'] = input_train_X['Out_of_Total'].astype(float)
    input_train_X['Rank'] = input_train_X['Rank'].astype(float)
    input_train_X = input_train_X.astype({col: 'int32' for col in input_train_X.select_dtypes('bool').columns})

    # weighted processing
    X_weighted = input_df_W.drop(["LinScore", "BinaryScore", "Weight"], axis=1)
    y_weighted = input_df_W["LinScore"]
    X_weighted['Out_of_Total'] = X_weighted['Out_of_Total'].astype(float)
    X_weighted['Rank'] = X_weighted['Rank'].astype(float)
    X_weighted = X_weighted.astype({col: 'int32' for col in X_weighted.select_dtypes('bool').columns})

    return input_train_X, input_train_y, X_weighted, y_weighted


# ex call
# input_train_X,input_train_y,X_weighted,y_weighted,columns_not_to_scale=preprocess_train_and_val_data_for_scalers(
#         train_folds_df_A_post_norm_df,ordinal_features_not_to_weight, columns_not_to_scale)


def preprocess_train_and_val_data(
        input_train_df,
        ordinal_features_not_to_weight,
        input_val_df,
        columns_not_to_scale):
    """This fx takes 2  input df ('input_train_df' a normal train_df, just with the weight parameter added, and 'input_val_df' and after 
    processing them all the same way, returns 6 df (x and y)"""

    input_df_W = input_train_df.drop(["Name", 'Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term'], axis=1)
    input_train_df = input_train_df.drop(["Name", 'Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term'], axis=1)
    input_val_df = input_val_df.drop(["Name", 'Original_PFP_Pred_GO_term', 'Updated_PFP_Pred_GO_term'], axis=1)
    features_labels = [x for x in list(input_df_W.columns) if x not in ordinal_features_not_to_weight]
    weights_col = "Weight"
    input_df_W[features_labels] = input_df_W[features_labels].astype(float)
    features_matrix = input_train_df[features_labels].values  # Shape: (n_samples, n_features)
    weights_vector = np.sqrt(input_df_W[weights_col].values)  # Shape: (n_samples,)

    # Reshape the weights vector to broadcast properly
    weights_matrix = weights_vector[:, np.newaxis]  # Shape: (n_samples, 1)    
    # Perform element-wise multiplication
    weighted_features = features_matrix.astype(float) * weights_matrix.astype(float)  # Shape: (n_samples, n_features)  

    # Assign the result back to the DataFrame
    input_df_W[features_labels] = weighted_features

    # train's processing
    input_train_X = input_train_df.drop(["LinScore", "BinaryScore", "Weight"], axis=1)
    input_train_y = input_train_df["LinScore"]
    input_train_X['Out_of_Total'] = input_train_X['Out_of_Total'].astype(float)
    input_train_X['Rank'] = input_train_X['Rank'].astype(float)
    input_train_X = input_train_X.astype({col: 'int32' for col in input_train_X.select_dtypes('bool').columns})

    # validation processing
    input_val_X = input_val_df.drop(["LinScore", "BinaryScore"], axis=1)
    input_val_y = input_val_df["LinScore"]
    input_val_X['Out_of_Total'] = input_val_X['Out_of_Total'].astype(float)
    input_val_X['Rank'] = input_val_X['Rank'].astype(float)
    input_val_X = input_val_X.astype({col: 'int32' for col in input_val_X.select_dtypes('bool').columns})

    # weighted processing
    X_weighted = input_df_W.drop(["LinScore", "BinaryScore", "Weight"], axis=1)
    y_weighted = input_df_W["LinScore"]
    X_weighted['Out_of_Total'] = X_weighted['Out_of_Total'].astype(float)
    X_weighted['Rank'] = X_weighted['Rank'].astype(float)
    X_weighted = X_weighted.astype({col: 'int32' for col in X_weighted.select_dtypes('bool').columns})

    return input_train_X, input_train_y, input_val_X, input_val_y, X_weighted, y_weighted, columns_not_to_scale


def read_reweight_and_preprocess_folds(
        val_fold_list,
        train_fold_list,
        num,
        number_chosen_bins,
        ordinal_features_not_to_weight,
        columns_not_to_scale,
        model_list,
        var1_weight_status
):
    """
    This fx uses the 5 CV to populate a dict, that I can use as the input to the next fx. The 5 CV are really the same, but I wanted
    to have them be in a dict to be pre populated, rather than a)reading b)weighting (in case of wls), c) and then changing the data types
    """

    tmp_train, tmp_val, tmp_weighted = [], [], []  # original_train_data_with_weights_column_added was df_with_weights
    # Part1: using a single C and single V (inputed as a list to 'calculate_weights' fx), generate 6 outputs and save them to 3 lists
    for (each_train, each_val) in zip(train_fold_list, val_fold_list):
        # each_train_df_with_weight_col, each_test_df_with_weight_col,original_train_data_with_weights_column_added=calculate_weights(each_train,each_val,number_chosen_bins) #previously was named, 'train_fold_df_A_post_norm_df'
        original_train_data_with_weights_column_added = calculate_weights(each_train, each_val,
                                                                          number_chosen_bins)  # previously was named, 'train_fold_df_A_post_norm_df'

        input_train_X, input_train_y, input_val_X, input_val_y, X_weighted, y_weighted, columns_not_to_scale = preprocess_train_and_val_data(
            original_train_data_with_weights_column_added,
            ordinal_features_not_to_weight,
            each_val,
            columns_not_to_scale
        )

        tmp_train.append([input_train_X, input_train_y])
        tmp_weighted.append([X_weighted, y_weighted])
        tmp_val.append([input_val_X, input_val_y])

    # Part2: Each of the tmp_train/tmp_weighted/tmp_val is of len(5), and each unit is of len2
    # Iterate through the 5 units and pull the X ([0], and y[1], and make them separate lists)
    train_folds_list_X_list = [preprocess_data_for_gpu1(sublist[0], columns_not_to_scale) for sublist in tmp_train]
    weight_folds_list_X_list = [preprocess_data_for_gpu1(sublist[0], columns_not_to_scale) for sublist in tmp_weighted]
    val_folds_list_X_list = [preprocess_data_for_gpu1(sublist[0], columns_not_to_scale) for sublist in tmp_val]
    train_folds_list_y_list = [sublist[1] for sublist in tmp_train]
    weight_folds_list_y_list = [sublist[1] for sublist in tmp_weighted]
    val_folds_list_y_list = [sublist[1] for sublist in tmp_val]

    # Part3: Add those lists to the dict of models
    dict_folds_unweighted, dict_folds_weighted = {}, {}
    for model in model_list:
        dict_folds_unweighted[model], dict_folds_weighted[model] = {}, {}
        dict_folds_unweighted[model] = {"trainUW": {}, "valUW": {}}
        dict_folds_weighted[model] = {"trainW": {}, "trainUW": {}, "valUW": {}}

        dict_folds_unweighted[model]["trainUW"]["features"] = train_folds_list_X_list
        dict_folds_unweighted[model]["trainUW"]["labels"] = train_folds_list_y_list
        dict_folds_unweighted[model]["valUW"]["features"] = val_folds_list_X_list
        dict_folds_unweighted[model]["valUW"]["labels"] = val_folds_list_y_list

        dict_folds_weighted[model]["trainW"]["features"] = weight_folds_list_X_list
        dict_folds_weighted[model]["trainW"]["labels"] = weight_folds_list_y_list
        dict_folds_weighted[model]["trainUW"]["features"] = train_folds_list_X_list
        dict_folds_weighted[model]["trainUW"]["labels"] = train_folds_list_y_list
        dict_folds_weighted[model]["valUW"]["features"] = val_folds_list_X_list
        dict_folds_weighted[model]["valUW"]["labels"] = val_folds_list_y_list

    if var1_weight_status == "weighted":
        return dict_folds_weighted
    else:
        return dict_folds_unweighted


def generate_hyperparam_configs(logger, list_input_alphas):
    # this fx is specifically for the linear regression models. I put this in, when I was doing Linear Regression, but for the rest, the params will be in the py files
    # Define baseline configuration
    baseline_config = {
        "eta0": 0.001,  # Default initial learning rate
        "batch_size": 32,  # Default batch size
        "epochs": 1000,  # Number of iterations over the dataset
        "tol": 0.001,  # Convergence tolerance
        "learning_rate": "constant",  # Default learning rate strategy
        "alpha": 0.0,  # No regularization (pure SGD)
        "n_iter_no_change": 5,  # Number of iterations with no improvement to stop early
        "l1_ratio": 0.15,  # Only used for ElasticNet (ignored in pure SGD)
        "fit_intercept": True,  # Whether to fit an intercept term
        "shuffle": True,  # Shuffle data before each epoch
        "verbose": True,  # No verbose logging
        "output_type": None  # Default output format

    }

    expanded_hyperparams = {
        "eta0": [0.00001, 0.0001, 0.001],
        # "eta0": [0.0001],
        "batch_size": [32, 128, 256],
        # "tol": [0.001],
        "tol": [0.0001, 0.001, 0.01],
        "epochs": [1000],
        "learning_rate": ["adaptive", "constant"],
        # "learning_rate": ["constant"],
        "n_iter_no_change": [5],
        "alpha": [0.00]

    }

    hyperparameter_options = {
        "ols": {"baseline_only": True},  # OLS has no hyperparameters
        "mb_SGD": {  # Grid search with base hyperparameters
            "baseline_only": False,
            **copy.deepcopy(expanded_hyperparams)
        },
        "lasso": {
            "baseline_only": False,
            **copy.deepcopy(expanded_hyperparams),
            "alpha": list_input_alphas["alpha"],
        },
        "ridge": {
            "baseline_only": False,
            **copy.deepcopy(expanded_hyperparams),
            "alpha": list_input_alphas["alpha"],
        },
        "elasticnet": {
            "baseline_only": False,
            **{**copy.deepcopy(expanded_hyperparams),
               "alpha": list_input_alphas["alpha"],
               "l1_ratio": [0.1, 0.5, 0.9]}
        }
    }

    # Generate hyperparameter combinations for each model
    organized_hyperparams = {}
    for model, params in hyperparameter_options.items():
        if params.get("baseline_only", False):
            organized_hyperparams[model] = {}
        else:
            # Generate combinations of hyperparameters
            keys, values = zip(*{k: v for k, v in params.items() if k != "baseline_only"}.items())
            combinations = [dict(zip(keys, v)) for v in product(*values)]
            organized_hyperparams[model] = {f"{model}_hp_{i + 1}": combo for i, combo in enumerate(combinations)}

    # Print the number of combinations generated for each model
    for model in organized_hyperparams:
        logger.info(
            f"model is {model} Generated {len(organized_hyperparams[model])} hyperparameter combinations for {model}.")

    # Add baseline configuration to models that require it
    for model in ["lasso", "mb_SGD", "ridge", "elasticnet"]:
        organized_hyperparams[model][f"{model}_hp_0"] = copy.deepcopy(
            baseline_config)  # stops changes in all models, when i made change in one odel

    # Print the number of combinations generated for each model
    for model in organized_hyperparams:
        logger.info(
            f"model is {model} Generated {len(organized_hyperparams[model])} hyperparameter combinations with the 0th  for {model}.")

    # setting for the hp_0: fully default
    organized_hyperparams["mb_SGD"]["mb_SGD_hp_0"]["alpha"] = 0.0
    organized_hyperparams["lasso"]["lasso_hp_0"]["alpha"] = 0.0001
    organized_hyperparams["ridge"]["ridge_hp_0"]["alpha"] = 0.0001
    organized_hyperparams["elasticnet"]["elasticnet_hp_0"]["alpha"] = 0.0001  #
    organized_hyperparams["elasticnet"]["elasticnet_hp_0"]["l1_ratio"] = 0.15

    return organized_hyperparams


def preprocess_data_for_gpu1(
        input_df_or_series
        , columns_not_to_scale
):
    """
    Preprocess the DataFrame to ensure compatibility with GPU processing by changing flaot64 to float32, and int64 to int32.
    """

    if isinstance(input_df_or_series, pd.Series):
        input_df_or_series = input_df_or_series.to_frame()

    for col in input_df_or_series.select_dtypes(include=['object']).columns:
        if col not in columns_not_to_scale:
            input_df_or_series.loc[:, col] = input_df_or_series[col].astype('float32')
    # Convert boolean columns to integers (bool -> int32)
    for col in input_df_or_series.select_dtypes(include=['bool']).columns:
        input_df_or_series.loc[:, col] = input_df_or_series[col].astype('float32')
    # Convert all floats to float32 for better GPU performance

    for col in input_df_or_series.columns:
        if isinstance(input_df_or_series[col].dtype, pd.SparseDtype):
            if col not in columns_not_to_scale:
                dense_col = input_df_or_series[col].sparse.to_dense()
                sparse_col = pd.arrays.SparseArray(dense_col, dtype='float32')
                input_df_or_series[col] = sparse_col
        elif input_df_or_series[col].dtype == 'float64' or input_df_or_series[col].dtype == 'int64':
            if col not in columns_not_to_scale:
                input_df_or_series.loc[:, col] = input_df_or_series[col].astype('float32')

    # print("Data preprocessing completed.")
    return input_df_or_series


def preprocess_data_for_gpu2(model, fold, best_scaler_fx, input_dic, columns_not_to_scale, var1_weight_status):
    # no cuml conversion
    """
    This function takes the model, fold, best scaler, the feature columsn and the dictionary of CV generated and it runs it throught 
    the column_transformer, paying attention to the WLS and using .fit_transform only on the NON weighted Data in all instances

    WLS and non WLS needs to be processed separately. WLS val data is transformed after the column transformer is fit on the WLS 'unweighted_data'
    
    """
    input_data_for_columns = list(input_dic[model]["trainUW"]["features"][0].columns)

    if var1_weight_status == "weighted":
        input_X_train_weighted = input_dic[model]["trainW"]["features"][fold]  # Weighted
        feature_columns = [col for col in input_X_train_weighted if col not in columns_not_to_scale]

        input_y_train = input_dic[model]["trainW"]["labels"][fold]  # Weighted

        input_X_train_unweighted = input_dic[model]["trainUW"]["features"][fold]

        input_X_val_unweighted = input_dic[model]["valUW"]["features"][fold]
        input_y_val = input_dic[model]["valUW"]["labels"][fold]

        column_transformer = ColumnTransformer(
            transformers=[("scaler", best_scaler_fx, feature_columns)],
            remainder="passthrough"
        )

        # Scaling training and transformed data: for the WLS:Fitting and Transforming
        scaled_array_weighted = column_transformer.fit_transform(input_X_train_weighted)
        scaled_train_df = pd.DataFrame(scaled_array_weighted, columns=input_X_train_weighted.columns)  # put under z
        # common name

        # initiate the column transformer on the train not the weight data!!
        column_transformer = ColumnTransformer(
            transformers=[("scaler", best_scaler_fx, feature_columns)],
            remainder="passthrough"
        )

        scaled_val_array = column_transformer.fit(input_X_train_unweighted).transform(input_X_val_unweighted)
        scaled_val_df = pd.DataFrame(scaled_val_array, columns=input_X_val_unweighted.columns)

        return scaled_train_df, input_y_train, scaled_val_df, input_y_val

    else:
        input_X_train_unweighted = input_dic[model]["trainUW"]["features"][fold]
        feature_columns = [col for col in input_X_train_unweighted if col not in columns_not_to_scale]
        input_y_train = input_dic[model]["trainUW"]["labels"][fold]
        input_X_val_unweighted = input_dic[model]["valUW"]["features"][fold]
        input_y_val = input_dic[model]["valUW"]["labels"][fold]

        column_transformer = ColumnTransformer(
            transformers=[("scaler", best_scaler_fx, feature_columns)],
            remainder="passthrough"
        )

        # Scaling training data (not weighted):Fitting and Transforming
        scaled_array = column_transformer.fit_transform(input_X_train_unweighted)
        scaled_train_df = pd.DataFrame(scaled_array, columns=input_X_train_unweighted.columns)

        # Scaling training data (not weighted):Fitting and Transforming
        scaled_val_array = column_transformer.transform(input_X_val_unweighted)
        scaled_val_df = pd.DataFrame(scaled_val_array, columns=input_X_val_unweighted.columns)
        return scaled_train_df, input_y_train, scaled_val_df, input_y_val


def preprocess_data_for_gpu3(
        best_scaler_is_log,
        model,
        num_folds,
        input_dic,
        columns_not_to_scale,
        model_dict,
        scalers,
        var1_weight_status
):
    """
    This fx will iterate through models and through folds and generate the GPU ready dataframes. Since I am only running one model at a time, the dict will 
    always have a key of 1. 
    """

    dict_cv_data_cpu = {}
    # for model in model_list:
    dict_cv_data_cpu[model] = {}
    best_scaler_fx = scalers[best_scaler_is_log]

    for fold_idx in range(num_folds):
        dict_cv_data_cpu[model][fold_idx] = {}
        scaled_train_X_df, unscaled_train_y_df, scaled_val_X_val, unscaled_val_y_df = preprocess_data_for_gpu2(
            model, fold_idx, best_scaler_fx, input_dic, columns_not_to_scale, var1_weight_status
        )
        dict_cv_data_cpu[model][fold_idx]["scaled_train_X"] = scaled_train_X_df
        dict_cv_data_cpu[model][fold_idx]["unscaled_train_y"] = unscaled_train_y_df
        dict_cv_data_cpu[model][fold_idx]["scaled_val_X"] = scaled_val_X_val
        dict_cv_data_cpu[model][fold_idx]["unscaled_val_y"] = unscaled_val_y_df

    return dict_cv_data_cpu


def preprocess_data_for_gpu3_5(
        logger,
        best_scaler_is_standard,
        model,
        num_folds,
        input_dic,
        columns_not_to_scale,
        model_dict,
        scalers,
        var1_weight_status
):
    """
    This fx will iterate through models and through folds and generate the GPU ready dataframes. Since I am only running one model at a time, the dict will 
    always have a key of 1. 
    """

    dict_cv_data_cpu = {}
    dict_cv_data_cpu[model] = {}
    best_scaler_fx = scalers[best_scaler_is_standard]
    for fold_idx in range(num_folds):
        dict_cv_data_cpu[model][fold_idx] = {}
        training_block = input_dic[model][fold_idx]['scaled_train_X']
        val_or_testing_block = input_dic[model][fold_idx]['scaled_val_X']

        feature_columns = [col for col in training_block.columns if col not in columns_not_to_scale]
        column_transformer = ColumnTransformer(
            transformers=[(best_scaler_is_standard, scalers[best_scaler_is_standard], feature_columns)],
            remainder="passthrough"
        )

        training_block2 = column_transformer.fit_transform(training_block)
        training_block3 = pd.DataFrame(training_block2, columns=training_block.columns)
        val_or_testing_block2 = column_transformer.transform(val_or_testing_block)
        val_or_testing_block3 = pd.DataFrame(val_or_testing_block2, columns=val_or_testing_block.columns)

        dict_cv_data_cpu[model][fold_idx]["scaled_train_X"] = training_block3
        dict_cv_data_cpu[model][fold_idx]["unscaled_train_y"] = input_dic[model][fold_idx]["unscaled_train_y"]
        dict_cv_data_cpu[model][fold_idx]["scaled_val_X"] = val_or_testing_block3
        dict_cv_data_cpu[model][fold_idx]["unscaled_val_y"] = input_dic[model][fold_idx]["unscaled_val_y"]

    return dict_cv_data_cpu


def preprocess_data_for_gpu4(dict_cv_data_cpu, model, fold):
    """
    This changes cpu df to the cudf df"""

    scaled_train_X = dict_cv_data_cpu[model][fold]['scaled_train_X']
    scaled_train_cudf = cudf.DataFrame.from_pandas(scaled_train_X)

    train_y = dict_cv_data_cpu[model][fold]['unscaled_train_y']
    y_cudf = cudf.Series(train_y)

    scaled_val_X = dict_cv_data_cpu[model][fold]['scaled_val_X']
    scaled_val_cudf = cudf.DataFrame.from_pandas(scaled_val_X)

    val_y = dict_cv_data_cpu[model][fold]["unscaled_val_y"]
    val_y_cudf = cudf.Series(val_y)

    return scaled_train_cudf, y_cudf, scaled_val_cudf, val_y_cudf


def alpha_selection_fx(
        logger,
        alphas_list,
        dict_assist,
        number_chosen_bins,
        grid_search_status,
        var2_class,
        var1_weight_status,
        output_out_dir,
        scalers,
        model,
        dict_cv_data_cpu,
        best_scaler_var,
        columns_not_to_scale,
        num_folds=5
):
    """
    Processes a single fold for a given model, computes metrics, and returns results as a DataFrame.
    """
    final_results = []
    current_model = ""
    for fold_idx in range(num_folds):
        if dict_assist != "Part4_of_5":
            print("problem detected")

        scaled_train_cudf, y_train_cudf, scaled_val_cudf, y_val_cudf = preprocess_data_for_gpu4(dict_cv_data_cpu, model,
                                                                                                fold_idx)
        df_test_with_weights = calculate_weights_mse(y_val_cudf, number_chosen_bins)
        df_train_with_weights = calculate_weights_mse(y_train_cudf, number_chosen_bins)

        for alpha_value in alphas_list:
            try:
                if model == "lasso":
                    current_model = cumlMBSGDRegressor(penalty="l1", alpha=alpha_value)
                    current_model.fit(scaled_train_cudf, y_train_cudf)
                    logger.info(
                        f"Running alpha selection for the: {model} for index: {fold_idx} and the model's parameters are: {current_model.get_params()}")
                    print("an alpha value is done on default param")

                elif model == "ridge":
                    current_model = cumlMBSGDRegressor(penalty="l2", alpha=alpha_value)
                    current_model.fit(scaled_train_cudf, y_train_cudf)
                    logger.info(
                        f"Running alpha selection for the: {model} for index: {fold_idx} and the model's parameters are: {current_model.get_params()}")


                elif model == "elasticnet":
                    current_model = cumlMBSGDRegressor(penalty="elasticnet", alpha=alpha_value)
                    current_model.fit(scaled_train_cudf, y_train_cudf)
                    logger.info(
                        f"Running alpha selection for the: {model} for index: {fold_idx} and the model's parameters are: {current_model.get_params()}")

                predictions_val = current_model.predict(scaled_val_cudf)
                predictions_train = current_model.predict(scaled_train_cudf)

                if model != "wls":

                    # Convert data to float64 explicitly
                    # y_val = y_val_cudf.to_numpy(dtype=np.float64)
                    y_pred_val = predictions_val.to_numpy(dtype=np.float64)
                    # y_train = y_train_cudf.to_numpy(dtype=np.float64)
                    # y_pred_train = predictions_train.to_numpy(dtype=np.float64)
                    # sample_weights_test = df_test_with_weights["Weight"]
                    # sample_weights_train = df_train_with_weights["Weight"]

                    overflow_threshold = 1e150

                    # Check for extreme values before computing MSE
                    # max_y_true = np.max(np.abs(y_true))
                    max_y_pred = np.max(np.abs(y_pred_val))

                    if max_y_pred > overflow_threshold or np.isnan(y_pred_val).any() or np.isinf(y_pred_val).any():
                        raise ValueError(
                            f"Overflow detected: Max y_pred={max_y_pred}, Min y_pred={np.min(y_pred_val)}. Skipping this model run.")

                    # Weighted MSE and R²
                    train_wmse = mean_squared_error(y_train_cudf.to_numpy(), predictions_train.to_numpy(),
                                                    sample_weight=df_train_with_weights["Weight"])
                    train_wr2 = r2_score(y_train_cudf.to_numpy(), predictions_train.to_numpy(),
                                         sample_weight=df_train_with_weights["Weight"])
                    train_mse = mean_squared_error(y_train_cudf.to_numpy(), predictions_train.to_numpy())
                    train_r2 = r2_score(y_train_cudf.to_numpy(), predictions_train.to_numpy())

                    val_wmse = mean_squared_error(y_val_cudf.to_numpy(), predictions_val.to_numpy(),
                                                  sample_weight=df_test_with_weights["Weight"])
                    val_wr2 = r2_score(y_val_cudf.to_numpy(), predictions_val.to_numpy(),
                                       sample_weight=df_test_with_weights["Weight"])
                    val_mse = mean_squared_error(y_val_cudf.to_numpy(), predictions_val.to_numpy())
                    val_r2 = r2_score(y_val_cudf.to_numpy(), predictions_val.to_numpy())

                    results_dict = {
                        "Model": model,
                        "Scaler": best_scaler_var,
                        "Alpha_value": alpha_value or np.NaN,
                        "Fold": fold_idx,
                        "Weighted Train MSE": train_wmse,
                        "Weighted Train R^2": train_wr2,
                        "Train MSE": train_mse,
                        "Train R^2": train_r2,
                        "Weighted Validation MSE": val_wmse,
                        "Weighted Validation R^2": val_wr2,
                        "Validation MSE": val_mse,
                        "Validation R^2": val_r2,
                        "Weight_Status": var1_weight_status,
                    }


            except Exception as e:
                print(f"Error processing model {model} in **alpha selection fx** for alpha_value {alpha_value}: {e}")
                logging.error(
                    f"Error processing model {model} in **alpha selection fx** for alpha_value {alpha_value}: {e}")
                w_mse, w_r2 = float('nan'), float('nan')
                val_mse, val_r2 = float('nan'), float('nan')

                results_dict = {
                    "Model": model,
                    "Scaler": best_scaler_var,
                    "Alpha_value": alpha_value or np.NaN,
                    "Fold": fold_idx,
                    "Weighted Validation MSE": w_mse,
                    "Weighted Validation R^2": w_r2,
                    "Validation MSE": val_mse,
                    "Validation R^2": val_r2,
                    "Weight_Status": var1_weight_status
                }

            final_results.append(results_dict)

    return pd.DataFrame(final_results)


def process_fold_new(logger,
                     dict_assist,
                     number_chosen_bins,
                     grid_search_status,
                     var2_class,
                     var1_weight_status,
                     output_dir_files,
                     index,
                     model,
                     scaled_train_cudf,
                     y_train_cudf,
                     scaled_val_cudf,
                     y_val_cudf,
                     best_scaler,
                     params,
                     params_key=None):
    """
    Processes a single fold for a given model, computes metrics, and returns results as a DataFrame.
    """
    logger.info(f"Global Orientation: Processing fold {index} for model: {model} with params: {params_key}...")

    try:
        df_test_with_weights = calculate_weights_mse(y_val_cudf, number_chosen_bins)
        df_train_with_weights = calculate_weights_mse(y_train_cudf, number_chosen_bins)

        if model == "ols":
            current_model = LinearRegression(copy_X=True)
            current_model.fit(scaled_train_cudf, y_train_cudf)
            stopping_epoch = "None"
            model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
            joblib.dump(current_model, model_filename)
            logger.info(f"fitted model: {model} for index: {index} without hyper-parameters is saved")
        elif model == "wls":  # manual weeights
            y_train_cudf = y_train_cudf.to_pandas()
            scaled_train_cudf = scaled_train_cudf.to_pandas()
            scaled_val_cudf = scaled_val_cudf.to_pandas()

            # Step 1: Fit initial OLS model to estimate residuals & save it. Hard-code the name of the model
            # weighted MSE
            current_model = sm.OLS(y_train_cudf, sm.add_constant(scaled_train_cudf)).fit()
            stopping_epoch = "None"
            predictions_ols_sm_train = current_model.predict(sm.add_constant(scaled_train_cudf))
            train_wmse = mean_squared_error(y_train_cudf.to_numpy(), predictions_ols_sm_train.to_numpy(),
                                            sample_weight=df_train_with_weights["Weight"])
            train_wr2 = r2_score(y_train_cudf.to_numpy(), predictions_ols_sm_train.to_numpy(),
                                 sample_weight=df_train_with_weights["Weight"])
            train_mse = mean_squared_error(y_train_cudf.to_numpy(), predictions_ols_sm_train.to_numpy())
            train_r2 = r2_score(y_train_cudf.to_numpy(), predictions_ols_sm_train.to_numpy())

            predictions_ols_sm_val = current_model.predict(sm.add_constant(scaled_val_cudf))
            val_wmse = mean_squared_error(y_val_cudf.to_numpy(), predictions_ols_sm_val.to_numpy(),
                                          sample_weight=df_test_with_weights["Weight"])
            val_wr2 = r2_score(y_val_cudf.to_numpy(), predictions_ols_sm_val.to_numpy(),
                               sample_weight=df_test_with_weights["Weight"])
            val_mse = mean_squared_error(y_val_cudf.to_numpy(), predictions_ols_sm_val.to_numpy())
            val_r2 = r2_score(y_val_cudf.to_numpy(), predictions_ols_sm_val.to_numpy())

            wls_result1 = {
                "Model": ["ols_stat_models"],
                "Scaler": [best_scaler],
                "Param": [params_key or np.NaN],
                "Fold": [index],
                "Weighted Train MSE": [train_wmse],
                "Weighted Train R^2": [train_wr2],
                "Train MSE": [train_mse],
                "Train R^2": [train_r2],
                "Weighted Validation MSE": [val_wmse],
                "Weighted Validation R^2": [val_wr2],
                "Validation MSE": [val_mse],
                "Validation R^2": [val_r2],
                "Weight_Status": [var1_weight_status],
                "Sample Weight": ["None yet: OLS"]
            }

            # Step 2: Compute residuals to get the WLS and save it as well
            predictions_res = current_model.predict(sm.add_constant(scaled_train_cudf))

            # Compute different versions of weights
            residuals = y_train_cudf - predictions_res
            # 1. Statsmodels suggested weights (based on predicted values deviation from mean)
            # weights = 1 / ((predictions_res - predictions_res.mean()) ** 2 + 1e-6)

            # 2. Residual-based weights (inverse of squared residuals)
            weights = 1 / ((residuals ** 2) + 1e-6)  # Adding a small value to avoid division by zero

            # # 3. Absolute residuals version (less sensitive to outliers)
            # weights = 1 / (np.abs(residuals) + 1e-6)

            # # 4. Log-transformed residuals (to smooth extreme values)
            # weights = 1 / (np.log1p(np.abs(residuals)) + 1e-6)  # log1p avoids log(0) issues

            # # # 5. 
            # weights = 1 / (np.sqrt(np.abs(residuals)) + 1e-6)  

            from statsmodels.stats.diagnostic import het_breuschpagan
            _, p_value, _, _ = het_breuschpagan(current_model.resid, sm.add_constant(scaled_train_cudf))
            print("Breusch-Pagan test p-value:", p_value)  # If p < 0.05, there's heteroscedasticity

            weights_cudf = pd.Series(weights)
            # Step 4: Fit the Weighted Least Squares (WLS) model using computed weights
            current_model = sm.WLS(y_train_cudf, sm.add_constant(scaled_train_cudf), weights=weights_cudf).fit()
            stopping_epoch = "None"
            logger.info("The shape of the **WLS"" adds 0th column as a intercept")
            # Step 5: Save the trained WLS model
            model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
            joblib.dump(current_model, model_filename)
            logger.info(f"fitted model: {model} for index: {index} without hyper-parameters is saved")
            # add an extra column
            scaled_val_cudf.insert(0, "intercept", 1)
            scaled_train_cudf.insert(0, "intercept", 1)
            logger.info(
                "The shape of the **WLS"" adds 0th column as a intercept, so an intercept column was added to the **scaled_val_cudf**")
            print(scaled_val_cudf.shape, scaled_train_cudf.shape)

            predictions_val = current_model.predict(scaled_val_cudf)
            predictions_train = current_model.predict(scaled_train_cudf)
            predictions_val.to_pickle(
                f"{output_dir_files}/predictions_val_fold_{index}_{var2_class}_{var1_weight_status}.pkl", protocol=4)
            are_different = not np.allclose(predictions_ols_sm_val, predictions_val, atol=1e-6)
            logger.info("Are OLS and WLS predictions different? {}".format(are_different))

            print(predictions_res[:10])
            print(predictions_val[:10])

            # Get stats
            train_wmse = mean_squared_error(y_train_cudf.to_numpy(), predictions_train.to_numpy(),
                                            sample_weight=df_train_with_weights["Weight"])
            train_wr2 = r2_score(y_train_cudf.to_numpy(), predictions_train.to_numpy(),
                                 sample_weight=df_train_with_weights["Weight"])
            train_mse = mean_squared_error(y_train_cudf.to_numpy(), predictions_train.to_numpy())
            train_r2 = r2_score(y_train_cudf.to_numpy(), predictions_train.to_numpy())

            val_wmse = mean_squared_error(y_val_cudf.to_numpy(), predictions_val.to_numpy(),
                                          sample_weight=df_test_with_weights["Weight"])
            val_wr2 = r2_score(y_val_cudf.to_numpy(), predictions_val.to_numpy(),
                               sample_weight=df_test_with_weights["Weight"])
            val_mse = mean_squared_error(y_val_cudf.to_numpy(), predictions_val.to_numpy())
            val_r2 = r2_score(y_val_cudf.to_numpy(), predictions_val.to_numpy())

            wls_result2 = {
                "Model": ["wls_stat_models"],
                "Scaler": [best_scaler],
                "Param": [params_key or np.NaN],
                "Fold": [index],
                "Weighted Train MSE": [train_wmse],
                "Weighted Train R^2": [train_wr2],
                "Train MSE": [train_mse],
                "Train R^2": [train_r2],
                "Weighted Validation MSE": [val_wmse],
                "Weighted Validation R^2": [val_wr2],
                "Validation MSE": [val_mse],
                "Validation R^2": [val_r2],
                "Weight_Status": [var1_weight_status],
                "Sample Weight": ["Manual: WLS"]
            }

            result_wls_combined = pd.concat([pd.DataFrame(wls_result1), pd.DataFrame(wls_result2)], axis=0)
            final_results = result_wls_combined
            return final_results

        elif model == "mb_SGD":
            params['epochs'] = 100
            current_model = cumlMBSGDRegressor(penalty="none", **params)
            current_model.fit(scaled_train_cudf, y_train_cudf)
            predictions_train = current_model.predict(scaled_train_cudf)
            predictions_val = current_model.predict(scaled_val_cudf)
            if predictions_val.isnull().sum().sum() > 0:
                print("The model is failing on the FIRST 100 epoch")
                logger.info(f"Model {model} has NA in the predictions after 100 epochs, so this parameter is skipped.")
                val_wmse, val_wr2 = float('nan'), float('nan')
                val_mse, val_r2 = float('nan'), float('nan')
                train_wmse, train_wr2 = float('nan'), float('nan')
                train_mse, train_r2 = float('nan'), float('nan')

                final_results = {
                    "Model": [model],
                    "Scaler": [best_scaler],
                    "Param": ["fail@Epoch100" + params_key],
                    "Fold": [index],
                    "Weighted Train MSE": [train_wmse],
                    "Weighted Train R^2": [train_wr2],
                    "Train MSE": [train_mse],
                    "Train R^2": [train_r2],
                    "Weighted Validation MSE": [val_wmse],
                    "Weighted Validation R^2": [val_wr2],
                    "Validation MSE": [val_mse],
                    "Validation R^2": [val_r2],
                    "Weight_Status": [var1_weight_status]
                }
                return pd.DataFrame(final_results)

            else:
                params['epochs'] = 1000
                current_model = cumlMBSGDRegressor(penalty="none", **params)
                logger.info(
                    f"Weights before **.fit()** for the fitted model: {model} for index: {index} are: {current_model.coef_} for Grid-Search-Status: {grid_search_status}")
                current_model.fit(scaled_train_cudf, y_train_cudf)
                logger.info(
                    f"Model Parameters for the fitted model: {model} for index: {index} are: {current_model.get_params()}")
                if dict_assist == "Part5_of_5":
                    model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
                    joblib.dump(current_model, model_filename)
                    logger.info(f"fitted model: {model} for index: {index} with best params is saved")

        elif model == "lasso":
            current_model = cumlMBSGDRegressor(penalty="l1", **params)
            logger.info(
                f"Weights before **.fit()** for the fitted model: {model} for index: {index} are: {current_model.coef_} for Grid-Search-Status: {grid_search_status}")
            current_model.fit(scaled_train_cudf, y_train_cudf)
            logger.info(
                f"Model Parameters for the fitted model: {model} for index: {index} are: {current_model.get_params()}")
            if dict_assist == "Part5_of_5":
                model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
                joblib.dump(current_model, model_filename)
                logger.info(f"fitted model: {model} for index: {index} with best params is saved")

        elif model == "ridge":
            current_model = cumlMBSGDRegressor(penalty="l2", **params)
            logger.info(
                f"Weights before **.fit()** for the fitted model: {model} for index: {index} are: {current_model.coef_} for Grid-Search-Status: {grid_search_status}")
            current_model.fit(scaled_train_cudf, y_train_cudf)
            logger.info(
                f"Model Parameters for the fitted model: {model} for index: {index} are: {current_model.get_params()}")
            if dict_assist == "Part5_of_5":
                model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
                joblib.dump(current_model, model_filename)
                logger.info(f"fitted model: {model} for index: {index} with best params is saved")

        elif model == "elasticnet":
            current_model = cumlMBSGDRegressor(penalty="elasticnet", **params)
            logger.info(
                f"Weights before **.fit()** for the fitted model: {model} for index: {index} are: {current_model.coef_} for Grid-Search-Status: {grid_search_status}")
            current_model.fit(scaled_train_cudf, y_train_cudf)
            logger.info(
                f"Model Parameters for the fitted model: {model} for index: {index} are: {current_model.get_params()}")
            if dict_assist == "Part5_of_5":
                model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
                joblib.dump(current_model, model_filename)
                logger.info(f"fitted model: {model} for index: {index} with best params is saved")

        logging.info(f"Only linear model saved to the hard drive")
        if model == "ols":
            scaled_val_cudf = scaled_val_cudf.to_pandas()

        predictions_train = current_model.predict(scaled_train_cudf)
        predictions_val = current_model.predict(scaled_val_cudf)

        if model != "wls":
            y_train = y_train_cudf.to_numpy(dtype=np.float64)
            y_pred_train = predictions_train.to_numpy(dtype=np.float64)

            y_val = y_val_cudf.to_numpy(dtype=np.float64)
            y_pred_val = predictions_val.to_numpy(dtype=np.float64)

            OVERFLOW_THRESHOLD = 1e150

            max_y_pred = np.max(np.abs(y_pred_val))
            if max_y_pred > OVERFLOW_THRESHOLD or np.isnan(y_pred_val).any() or np.isinf(y_pred_val).any():
                raise ValueError(
                    f"Overflow detected: Max y_pred={max_y_pred}, Min y_pred={np.min(y_pred_val)}. Skipping this model run.")
            # Weighted MSE and R²
            train_wmse = mean_squared_error(y_train, y_pred_train, sample_weight=df_train_with_weights["Weight"])
            train_wr2 = r2_score(y_train, y_pred_train, sample_weight=df_train_with_weights["Weight"])
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)

            val_wmse = mean_squared_error(y_val, y_pred_val, sample_weight=df_test_with_weights["Weight"])
            val_wr2 = r2_score(y_val, y_pred_val, sample_weight=df_test_with_weights["Weight"])
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)


    except Exception as e:
        print(f"Error processing model {model} with params {params_key}: {e}")
        logging.error(f"Error processing model {model} with params {params_key}: {e}")
        wmse, wr2 = float('nan'), float('nan')
        val_mse, val_r2 = float('nan'), float('nan')

    final_results = {
        "Model": [model],
        "Scaler": [best_scaler],
        "Param": [params_key or np.NaN],
        "Fold": [index],
        "Weighted Train MSE": [train_wmse],
        "Weighted Train R^2": [train_wr2],
        "Train MSE": [train_mse],
        "Train R^2": [train_r2],
        "Weighted Validation MSE": [val_wmse],
        "Weighted Validation R^2": [val_wr2],
        "Validation MSE": [val_mse],
        "Validation R^2": [val_r2],
        "Weight_Status": [var1_weight_status]
    }

    return pd.DataFrame(final_results)


def cv_with_best_scaler(
        logger,
        dict_assist,
        number_chosen_bins,
        df_test_with_weights,
        grid_search_status,
        var2_class,
        organized_hyperparams_dict,
        var1_weight_status,
        output_out_dir,
        scalers,
        model,
        dict_cv_data_cpu,
        best_scaler_var,
        columns_not_to_scale,
        num_folds=5
):
    """
    This fx will the sets up the ML training for the linear regression models.
    """
    list_stats = []
    dict_tmp = {}
    original_organized_hyperparams_dict = deepcopy(organized_hyperparams_dict)
    for fold_idx in range(num_folds):
        if dict_assist == "Part4":
            print("First rotation being done")
        else:

            organized_hyperparams_dict = edit_dict_best_parameters_per_fold(original_organized_hyperparams_dict,
                                                                            fold_idx, model)
            print("second rotation being done")

        inner_start_time = time.time()
        scaled_train_X_cudf, train_y_cudf, scaled_val_X_cudf, val_y_cudf = preprocess_data_for_gpu4(dict_cv_data_cpu,
                                                                                                    model, fold_idx)

        if model == "ols" or model == "wls":
            stats = process_fold_new(
                logger,
                dict_assist,
                number_chosen_bins,
                # df_test_with_weights,
                grid_search_status,
                var2_class,
                var1_weight_status,
                output_out_dir,
                index=fold_idx,
                model=model,
                scaled_train_cudf=scaled_train_X_cudf,
                y_train_cudf=train_y_cudf,
                scaled_val_cudf=scaled_val_X_cudf,
                y_val_cudf=val_y_cudf,
                best_scaler=best_scaler_var,
                params={},  # No hyperparameters applied
                params_key="default"
            )
            list_stats.append(stats)
        else:
            for params_key, params in organized_hyperparams_dict[model].items():
                stats = process_fold_new(
                    logger,
                    dict_assist,
                    number_chosen_bins,
                    # df_test_with_weights,
                    grid_search_status,
                    var2_class,
                    var1_weight_status,
                    output_out_dir,
                    index=fold_idx,
                    model=model,
                    scaled_train_cudf=scaled_train_X_cudf,
                    y_train_cudf=train_y_cudf,
                    scaled_val_cudf=scaled_val_X_cudf,
                    y_val_cudf=val_y_cudf,
                    best_scaler=best_scaler_var,
                    params=params,
                    params_key=params_key
                )
                list_stats.append(stats)

        elapsed_time_inner = time.time() - inner_start_time
        logger.info("Elapsed Time: ", elapsed_time_inner)

    summary_df = pd.concat(list_stats)
    return summary_df


def cv_with_gridsearch_for_linear_fx(
        logger,
        dict_assist,
        number_chosen_bins,
        grid_search_status,
        var2_class,
        organized_hyperparams_dict,
        var1_weight_status,
        output_out_dir,
        scalers,
        model,
        dict_cv_data_cpu,
        best_scaler_var,
        columns_not_to_scale,
        num_folds=5
):
    """
    This fx will the sets up the ML training for the linear regression models.
    """
    list_stats = []
    dict_tmp = {}

    original_organized_hyperparams_dict = deepcopy(organized_hyperparams_dict)
    for fold_idx in range(num_folds):
        if dict_assist == "Part4_of_5":
            logger.info(
                f"Sending in grid search parameters (doing grid search) dict for model: {model} for index: {fold_idx}")
        elif dict_assist == "Part5_of_5":
            organized_hyperparams_dict = edit_dict_best_parameters_per_fold(original_organized_hyperparams_dict,
                                                                            fold_idx, model)
            logger.info(
                f"Sending in pre-selected grid-search parameters (post grid-search) dict for model: {model} for index: {fold_idx}")
        elif dict_assist == "Part5":
            logger.info(f"Model: {model} for index: {fold_idx} has no hyper-parameters")

        inner_start_time = time.time()
        scaled_train_X_cudf, train_y_cudf, scaled_val_X_cudf, val_y_cudf = preprocess_data_for_gpu4(dict_cv_data_cpu,
                                                                                                    model, fold_idx)

        if model == "ols" or model == "wls":
            stats = process_fold_new(
                logger,
                dict_assist,
                number_chosen_bins,
                grid_search_status,
                var2_class,
                var1_weight_status,
                output_out_dir,
                index=fold_idx,
                model=model,
                scaled_train_cudf=scaled_train_X_cudf,
                y_train_cudf=train_y_cudf,
                scaled_val_cudf=scaled_val_X_cudf,
                y_val_cudf=val_y_cudf,
                best_scaler=best_scaler_var,
                params={},  # No hyperparameters applied
                params_key="default"
            )
            list_stats.append(stats)
        else:
            for params_key, params in organized_hyperparams_dict[model].items():
                print("params_key", params_key)
                stats = process_fold_new(
                    logger,
                    dict_assist,
                    number_chosen_bins,
                    grid_search_status,
                    var2_class,
                    var1_weight_status,
                    output_out_dir,
                    index=fold_idx,
                    model=model,
                    scaled_train_cudf=scaled_train_X_cudf,
                    y_train_cudf=train_y_cudf,
                    scaled_val_cudf=scaled_val_X_cudf,
                    y_val_cudf=val_y_cudf,
                    best_scaler=best_scaler_var,
                    params=params,
                    params_key=params_key
                )
                list_stats.append(stats)

    elapsed_time_inner = time.time() - inner_start_time

    summary_df = pd.concat(list_stats)
    return summary_df


def choosing_best_scaler_selection1(
        full_df_original,
        weight_status,
        var2_class
):
    """
    This fx post processes the big df generated in the previous step and returns the original scaler df and sorted scalers for each models
    """
    tmpL = []
    aggregg1 = full_df_original.groupby('Model Name').apply(
        lambda x: x[['Model Name', 'Scaler', 'Validation R²', 'Validation MSE']].sort_values(by=['Validation R²',
                                                                                                 'Validation MSE'],
                                                                                             ascending=[False, True]))
    aggregg1 = pd.DataFrame(aggregg1).reset_index(drop=True)
    groupings = aggregg1.groupby(['Model Name'], group_keys=True)
    for name, group in groupings:
        tmpL.append(group.head(1))
    top_average_scaler_df = pd.concat(tmpL).reset_index(drop=True)
    full_df_original["Weight Status"] = weight_status
    top_average_scaler_df["Downsampled"] = var2_class
    top_average_scaler_df["Weight Status"] = weight_status

    return full_df_original, top_average_scaler_df


def choose_best_param_per_fold_per_model(
        logger,
        model,
        input_df,
        var1,
        var2,
        var3,
        var4):
    """
    This fx chooses the best param in 3 ways:a) if there is a single param with best R2, and best MSE, if there is a 2+ params with equal R2/MSE, but one of them is the
    default (in this case, a default is chosen), and if htere are 2+ best params with the same R2/MSE, then choose the top one. I will log this case. 
    """

    return_dict_for_five_folds = {}
    return_dict_for_five_folds[model] = {}
    for fold_id in input_df[var1].drop_duplicates().to_list():
        tmp_df = input_df[input_df[var1] == fold_id]
        tmp_df_sorted = tmp_df.sort_values(by=[var2, var3], ascending=[False, True])

        top_row = tmp_df_sorted.iloc[0]
        duplicate_check_columns = [
            "Weighted Validation MSE",
            "Weighted Validation R^2",
            "Validation MSE",
            "Validation R^2"
        ]
        selected_row = tmp_df_sorted.iloc[0]
        selected_partial_row = selected_row[duplicate_check_columns]

        selected_partial_row_df = selected_partial_row.to_frame().T  # Convert Series to DataFrame and transpose
        tmp_df_dupl = tmp_df_sorted[tmp_df_sorted[duplicate_check_columns].eq(selected_partial_row.values).all(axis=1)]
        if tmp_df_dupl.shape[0] == 1:
            selected_row = top_row
            var_reason = "single best answer by r2/mse"
            var_shape = 1
            top_row_param_var = selected_row[var4]
        else:
            if tmp_df_dupl['Param'].str.contains('hp_0', case=False).any():
                selected_row = tmp_df_dupl[tmp_df_dupl['Param'].str.contains('hp_0')]
                top_row_param_var = list(selected_row[var4])[0]
                var_shape = tmp_df_dupl.shape[0]
                var_reason = "2+ best params, with default"
            else:
                selected_row = tmp_df_dupl.iloc[0]
                top_row_param_var = selected_row[var4]
                var_reason = "2+ best params, without default"
                var_shape = tmp_df_dupl.shape[0]
        return_dict_for_five_folds[model][fold_id] = [top_row_param_var, var_reason, var_shape]

    return return_dict_for_five_folds


def edit_dict_best_parameters_per_fold(input_dict_5folds, fold_idx, model_name):
    """
    This fx selects a specific fold from a model in the 5-fold dictionary, according to the fold currently being run
    through a **cv_with_best_scaler** fx.

    Parameters:
    - input_dict_5folds (dict): Dictionary containing model parameters for multiple folds.
    - fold_idx (int): The index of the fold to extract (must be 0, 1, 2, 3, or 4).
    - model_name (str): The model name (e.g., "lasso" or "ridge").

    Returns:
    - dict_best_parameters_per_fold_tmp (dict): Dictionary containing only the selected fold.
    """

    # Part1: Ensure fold_idx is strictly within {0, 1, 2, 3, 4}
    valid_fold_indices = {0, 1, 2, 3, 4}
    if fold_idx not in valid_fold_indices:
        raise ValueError(f"Invalid fold_idx: {fold_idx}. Allowed values: {valid_fold_indices}")

    # Part2: made an index of the folds in the **dict_best_parameters_per_fold**
    dict_fold_list = []
    for count in input_dict_5folds[model_name].keys():
        dict_fold_list.append(count)

    # Part3: Set up a new dict, with the same global name:
    tmp_dict = {}
    dict_model_name_var = list(input_dict_5folds.keys())[0]
    tmp_dict[dict_model_name_var] = {}

    # Part4: using the fold_idx input to select only the fold in which I am working with to assign the param_values to the dict_tmp
    tmp_dict[dict_model_name_var] = input_dict_5folds[model_name][dict_fold_list[fold_idx]]

    return tmp_dict


def generate_dict_with_best_chosen_hyper_param_per_model_per_fold(dict_best_param_key_per_fold,
                                                                  organized_hyperparams_dict):  # new
    """
    This function constructs a dictionary (`dict_best_parameters_per_fold`) that maps models and folds
    to the best hyperparameters.

    Parameters:
    - dict_best_param_key_per_fold: {model_name: {fold_id: best_param_key}}
    - organized_hyperparams_dict: {model_name: {param_key: param_value}}

    Returns:
    - dict_best_parameters_per_fold: {model_name: {fold_id: best_param_value}}
    """
    dict_best_parameters_per_fold = {}

    # Step 1: Generate the emphty dictionary structure: dict: model: fold
    dict_best_parameters_per_fold = {}
    for model, fold in dict_best_param_key_per_fold.items():
        dict_best_parameters_per_fold[model] = {}
        for fold_key in list(fold.keys()):
            fold_key = "fold_" + str(fold_key)
            dict_best_parameters_per_fold[model][fold_key] = {}

    # Step2: Generate the list of best params from the input dict, **dict_best_param_key_per_fold**
    list_input_params = []
    for param_key in dict_best_param_key_per_fold[model].values():
        list_input_params.append(param_key[0])

    # Step3: Fill the dict from step1 with param_key (that will be the new keys) from step2
    for key, values in dict_best_parameters_per_fold.items():
        for v1, param_key in zip(values.keys(), list_input_params):
            dict_best_parameters_per_fold[key][v1][param_key] = {}

    # Step4: Assign the newly filled dict with the values form the "organized_hyperparams_dict"
    for key, values in dict_best_parameters_per_fold.items():
        for fold_id, params in values.items():  # `params` should be a dictionary
            for param_key in params:  # Loop through the keys inside the param dictionary
                dict_best_parameters_per_fold[key][fold_id][param_key] = organized_hyperparams_dict[model][param_key]

    return dict_best_parameters_per_fold
