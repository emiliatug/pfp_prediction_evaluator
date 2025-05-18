import logging
from collections import defaultdict
from copy import deepcopy
from random import sample

import cudf as cudf
import cupy as cp
import joblib
from cuml import LinearRegression
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.linear_model import LinearRegression
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml.svm import LinearSVR
from cuml.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

cp.cuda.Device().use()
import warnings

warnings.filterwarnings("ignore",
                        message="pandas.DataFrame with sparse columns found")  # this just silences a SPECIFIC warnings

# Now import other libraries
import pandas as pd
import numpy as np


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
    print("y_labels_list dtype:", type(y_labels_list))
    print("y_labels_list first few values:", y_labels_list[:5])

    print("bins dtype:", type(bins))
    print("bins first few values:", bins[:5])
    # Compute bin indices
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
    df_with_weights_train_selc_columns_LinScore_Weight = df_with_weights_train[["LinScore", "Weight"]].drop_duplicates()
    # Merge weights into the original dataset
    original_train_data_with_weights_column_added = pd.merge(original_train_data,
                                                             df_with_weights_train_selc_columns_LinScore_Weight,
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


# ex call
# input_train_X,input_train_y,input_val_X,input_val_y,X_weighted,y_weighted,columns_not_to_scale=preprocess_train_and_val_data(
# train_df_post_norm_df,ordinal_features_not_to_weight,val_df,columns_not_to_scal)

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
                dense_col = input_df_or_series[col].sparse.to_dense().astype('float32')
                # sparse_col = pd.arrays.SparseArray(dense_col, dtype='float32')
                sparse_col = dense_col  # keep this line so the code matches the one in the linear fx

                input_df_or_series[col] = sparse_col
        elif input_df_or_series[col].dtype == 'float64' or input_df_or_series[col].dtype == 'int64':
            if col not in columns_not_to_scale:
                input_df_or_series.loc[:, col] = input_df_or_series[col].astype('float32')

    return input_df_or_series


def preprocess_data_for_gpu2(model, fold, best_scaler_fx, input_dic, columns_not_to_scale, var1_weight_status):
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

        input_X_train_unweighed = input_dic[model]["trainUW"]["features"][fold]

        input_X_val_unweighted = input_dic[model]["valUW"]["features"][fold]
        input_y_val = input_dic[model]["valUW"]["labels"][fold]

        column_transformer = ColumnTransformer(
            transformers=[("scaler", best_scaler_fx, feature_columns)],
            remainder="passthrough"
        )

        # Scaling training and transformed data: for the WLS:Fitting and Transforming
        scaled_array_weighted = column_transformer.fit_transform(input_X_train_weighted)
        scaled_train_df = pd.DataFrame(scaled_array_weighted, columns=input_X_train_weighted.columns)  # put under
        # common name

        # initiate the column transformer on the train not the weight data!!
        column_transformer = ColumnTransformer(
            transformers=[("scaler", best_scaler_fx, feature_columns)],
            remainder="passthrough"
        )

        scaled_val_array = column_transformer.fit(input_X_train_unweighed).transform(input_X_val_unweighted)
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
    best_scaler_fx = scalers[best_scaler_is_log[0]]
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


def process_fold_new_rf(logger,
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
                        #  best_scaler, #no scaler for random forest is needed
                        params,
                        params_key=None):
    """
    Processes a single fold for a given model, computes metrics, and returns results as a DataFrame.
    """
    logger.info(f"Global Orientation: Processing fold {index} for model: {model} with params: {params_key}...")

    if model == "random_forest":
        try:
            df_test_with_weights = calculate_weights_mse(y_val_cudf, number_chosen_bins)
            df_train_with_weights = calculate_weights_mse(y_train_cudf, number_chosen_bins)
            current_model = curfr(**params)
            current_model.fit(scaled_train_cudf, y_train_cudf)
            logger.info(f"Model info: dict_assist is  {dict_assist} and params_key is : {params_key}")
            logger.info(
                f"Model Parameters for the fitted model: {model} for index: {index} for Grid-Search-Status: {grid_search_status} are: {current_model.get_params()}")

            if dict_assist == "Part5_of_5":
                model_filename = f"{output_dir_files}/model_{model}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
                joblib.dump(current_model, model_filename)
                logger.info(f"fitted model: {model} for index: {index} with best params is saved")

            predictions_train = current_model.predict(scaled_train_cudf)
            predictions_val = current_model.predict(scaled_val_cudf)
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
            val_r2 = r2_score(y_val_cudf.to_numpy(dtype=np.float64), predictions_val.to_numpy(dtype=np.float64))

        except Exception as e:
            # pass
            print(f"Error processing model {model} with params {params_key}: {e}")
            logging.error(f"Error processing model {model} with params {params_key}: {e}")
            val_wmse, val_wr2 = float('nan'), float('nan')
            val_mse, val_r2 = float('nan'), float('nan')

        final_results = {
            "Model": [model],
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
            "Dict_Assist": [dict_assist],
        }

    return pd.DataFrame(final_results)


def cv_per_model_rf(
        RANDOM_STATE,
        scaler_dict,
        best_scaler_var,
        logger,
        dict_assist,
        number_chosen_bins,
        grid_search_status,
        var2_class,
        organized_hyperparams_dict,
        var1_weight_status,
        output_out_dir,
        model,
        dict_cv_data_cpu,
        columns_not_to_scale,
        num_folds=5
):
    """
    This fx will the sets up the ML training for the linear regression models.
    """
    # list_stats = []
    dict_tmp = {}
    if dict_assist == "Part4_of_5":
        list_stats = []
        var1 = "train"
        var2 = "val"
    elif dict_assist == "Part5_of_5":
        list_stats = []
        var1 = "train_val"
        var2 = "test"
    dict_tmp[var1] = {}
    dict_tmp[var2] = {}
    for fold_idx in range(num_folds):
        for each_key in dict_cv_data_cpu.keys():
            dict_cv_data_cpu[each_key] = dict_cv_data_cpu[each_key].fillna(0)
            if "CV" + str(fold_idx) in each_key and var1 in each_key:
                dict_cv_data_cpu[each_key]['Out_of_Total'] = dict_cv_data_cpu[each_key]['Out_of_Total'].astype(float)
                dict_cv_data_cpu[each_key]['Rank'] = dict_cv_data_cpu[each_key]['Rank'].astype(float)
                dict_tmp[var1][fold_idx] = dict_cv_data_cpu[each_key]
            elif "CV" + str(fold_idx) in each_key and var2 in each_key:
                dict_cv_data_cpu[each_key]['Out_of_Total'] = dict_cv_data_cpu[each_key]['Out_of_Total'].astype(float)
                dict_cv_data_cpu[each_key]['Rank'] = dict_cv_data_cpu[each_key]['Rank'].astype(float)
                dict_tmp[var2][fold_idx] = dict_cv_data_cpu[each_key]

    original_organized_hyperparams_dict = deepcopy(organized_hyperparams_dict)
    # instead of the manual gradient search, use the random gradient search WITHOUT replacement
    if dict_assist == "Part4_of_5" and grid_search_status == "Random-Search":
        # This is where I am changing the number of samples to be run in the random forest. I saw a number of 60 iterations
        list_all_keys = sample(list(organized_hyperparams_dict[model].keys()),
                               int(len(list(organized_hyperparams_dict[model].keys())) * float(0.1)))
        # logger.info(list_all_keys)   
        logger.info(
            f"Orgnized HyperParamater Dictionary Global Orientation: dict_assist and  grid_search_status are,{dict_assist},{grid_search_status},{len(list_all_keys)},{list_all_keys}")

        new_dict = {}
        new_dict[model] = {}
        for el in list_all_keys:
            new_dict[model][el] = organized_hyperparams_dict[model][el]
        organized_hyperparams_dict = new_dict

    for fold_idx in range(num_folds):
        # If initializing script is script_1, then ignore the following logic
        if dict_assist == "Part4_of_5":
            dict_to_add = pd.read_pickle(
                '/media/deep/DATA/PycharmProjects/Summer2024/FF_editing_transfer/E_Tugolukov/Global_Part2_ML_not_DL/Regression/Results/IntermediateData/Decision_Trees/Random_Forest/Part2_4/Initial_NotExpanded/dict_best_parameters_values_per_fold.pkl'
                )
            new_key = f"fold_{fold_idx}"
            random_forest_hp__1 = dict_to_add[model][new_key]
            first_key = list(dict_to_add[model][new_key].keys())[0]
            logger.info(f"before update {original_organized_hyperparams_dict[model].keys()}")
            random_forest_hp__1 = {'random_forest_hp__1': dict_to_add[model][new_key][first_key]}
            organized_hyperparams_dict['random_forest'].update(random_forest_hp__1)
            logger.info(f"after update {organized_hyperparams_dict[model].keys()}")
            logger.info(f"for fold after update fold_idx {organized_hyperparams_dict[model]}")
            logger.info(f"Checkpoint:Part4_of_5")
            logger.info(
                f"Sending in grid search parameters (doing grid search) dict for model: {model} for index: {fold_idx}")

        elif dict_assist == "Part5_of_5":
            logger.info("Checkpoint:Part4_of_5")
            print(original_organized_hyperparams_dict)
            organized_hyperparams_dict = edit_dict_best_parameters_per_fold(original_organized_hyperparams_dict,
                                                                            fold_idx, model)
            logger.info(
                f"Sending in pre-selected grid-search parameters (post grid-search) dict for model: {model} for index: {fold_idx}")

        logger.info("Feature Scaling is added here")

        logger.info(f"The scaler is: {best_scaler_var[0]}")
        if best_scaler_var[0] != "None":
            logger.info("Processing with the scaling the data")
            training_block = dict_tmp[var1][fold_idx].drop("LinScore", axis=1)
            val_or_testing_block = dict_tmp[var2][fold_idx].drop("LinScore", axis=1)

            feature_columns = [col for col in training_block.columns if col not in columns_not_to_scale]
            column_transformer = ColumnTransformer(
                transformers=[(best_scaler_var[0], scaler_dict[best_scaler_var[0]], feature_columns)],
                remainder="passthrough"
            )

            training_block2 = column_transformer.fit_transform(training_block)
            training_block3 = pd.DataFrame(training_block2, columns=training_block.columns)
            val_or_testing_block2 = column_transformer.transform(val_or_testing_block)
            val_or_testing_block3 = pd.DataFrame(val_or_testing_block2, columns=val_or_testing_block.columns)
            logger.info("Feature Scaling is added here")

            train_X_cudf = cudf.DataFrame.from_pandas(training_block3)
            train_y_cudf = cudf.Series(dict_tmp[var1][fold_idx]["LinScore"])

            val_X_cudf = cudf.DataFrame.from_pandas(val_or_testing_block3)
            val_y_cudf = cudf.Series(dict_tmp[var2][fold_idx]["LinScore"])
        else:
            logger.info("Processing with the NOT scaling the data")
            train_X_cudf = cudf.DataFrame.from_pandas(dict_tmp[var1][fold_idx].drop("LinScore", axis=1))
            train_y_cudf = cudf.Series(dict_tmp[var1][fold_idx]["LinScore"])

            val_X_cudf = cudf.DataFrame.from_pandas(dict_tmp[var2][fold_idx].drop("LinScore", axis=1))
            val_y_cudf = cudf.Series(dict_tmp[var2][fold_idx]["LinScore"])

        for params_key, params in organized_hyperparams_dict[model].items():
            stats = process_fold_new_rf(
                logger,
                dict_assist,
                number_chosen_bins,
                grid_search_status,
                var2_class,
                var1_weight_status,
                output_out_dir,
                index=fold_idx,
                model=model,
                scaled_train_cudf=train_X_cudf,
                y_train_cudf=train_y_cudf,
                scaled_val_cudf=val_X_cudf,
                y_val_cudf=val_y_cudf,
                params=params,
                params_key=params_key
            )
            list_stats.append(stats)

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


# new
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
                selected_row = tmp_df_dupl[tmp_df_dupl['Param'].str.contains('hp_25')]
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
        # print("count is ", count)
        dict_fold_list.append(count)

    # print("length of dict_fold_list: ", len(dict_fold_list))

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
                print(param_key)
                dict_best_parameters_per_fold[key][fold_id][param_key] = organized_hyperparams_dict[model][param_key]

    return dict_best_parameters_per_fold


# ex call
# dict_best_parameters_per_fold=generate_dict_with_best_chosen_hyper_param_per_model_per_fold("lasso",dict_best_param_key_per_fold,organized_hyperparams_dict)

def generate_hyperparam_configs_general(logger, expanded_hyperparams, model_dict, model_name):
    list_params_names = []
    organized_hyperparams_dict = {}
    organized_hyperparams_dict[model_name] = {}

    list_params = list(ParameterGrid(expanded_hyperparams))
    for each_value in range(1, len(list_params) + 1):
        list_params_names.append(f"{model_name}_hp_{each_value}")
    for each_param_name, each_param in zip(list_params_names, list_params):
        organized_hyperparams_dict[model_name][each_param_name] = each_param

    print(type(model_dict['random_forest']))
    default_params = model_dict[model_name].get_params()
    # default_params=model_dict[model_name]().get_params() #This worked for the run_global2_part4and5RF1.sh and run_global2_part4and5RF2.sh,but
    # when I tried to fix the 2.sh it stopped working
    default_params0 = {k: v for k, v in default_params.items() if k != 'handle'}
    # organized_hyperparams_dict[model_name]["random_forest_hp_0"]=default_params0
    organized_hyperparams_dict[model_name][f"{model_name}_hp_0"] = default_params0

    # Print the number of combinations generated for each model
    for model in organized_hyperparams_dict:
        logger.info(
            f"model is {model} Generated {len(organized_hyperparams_dict[model])} hyperparameter combinations for {model}.")

    return organized_hyperparams_dict


def cv_per_model_svr(
        model_dict,
        RANDOM_STATE,
        scaler_dict,
        best_scaler_var,
        logger,
        dict_assist,
        number_chosen_bins,
        grid_search_status,
        var2_class,
        organized_hyperparams_dict,
        var1_weight_status,
        output_out_dir,
        model,
        dict_cv_data_cpu,
        columns_not_to_scale,
        num_folds=5
):
    """
    This fx will the sets up the ML training for the linear regression models.
    """
    full_dict = deepcopy(organized_hyperparams_dict)
    dict_tmp = {}
    if dict_assist == "Part4_of_5":
        list_stats = []
        var1 = "train"
        var2 = "val"
    elif dict_assist == "Part5_of_5":
        list_stats = []
        var1 = "train_val"
        var2 = "test"
    dict_tmp[var1] = {}
    dict_tmp[var2] = {}
    for fold_idx in range(num_folds):
        for each_key in dict_cv_data_cpu[model][fold_idx].keys():
            if not isinstance(dict_cv_data_cpu[model][fold_idx][each_key], pd.Series):
                dict_cv_data_cpu[model][fold_idx][each_key]['Out_of_Total'] = \
                dict_cv_data_cpu[model][fold_idx][each_key]['Out_of_Total'].astype(float)
                dict_cv_data_cpu[model][fold_idx][each_key]['Rank'] = dict_cv_data_cpu[model][fold_idx][each_key][
                    'Rank'].astype(float)
                dict_tmp[var1][fold_idx] = dict_cv_data_cpu[model][fold_idx]
                dict_tmp[var2][fold_idx] = dict_cv_data_cpu[model][fold_idx]

    original_organized_hyperparams_dict = deepcopy(organized_hyperparams_dict)
    # instead of the manual gradient search, use the random gradient search WITHOUT replacement
    if dict_assist == "Part4_of_5" and grid_search_status == "Random-Search":
        # This is where I am changing the number of samples to be run in the random forest. I saw a number of 60 iterations
        list_all_keys = sample(list(organized_hyperparams_dict[model].keys()),
                               int(len(list(organized_hyperparams_dict[model].keys())) * float(0.198)))
        logger.info(
            f"Orgnized HyperParamater Dictionary Global Orientation: dict_assist and  grid_search_status are,{dict_assist},{grid_search_status},{len(list_all_keys)},{list_all_keys}")
        new_dict = {}
        new_dict[model] = {}
        for el in list_all_keys:
            new_dict[model][el] = organized_hyperparams_dict[model][el]
        organized_hyperparams_dict = new_dict
        logger.info(
            f"There are {len(organized_hyperparams_dict[model].keys())} in organized_hyperparams_dict after the Random-Search")

    for fold_idx in range(num_folds):
        # organized_hyperparams_dict=deepcopy(original_organized_hyperparams_dict)
        # print(organized_hyperparams_dict['svr_kernel_rbf'].keys()) #len is 163 (162 of the my own seleciton and 1 default)
        # If initializing script is script_1, then ignore the following logic
        if dict_assist == "Part4_of_5" and grid_search_status == "Grid-Search":
            organized_hyperparams_dict = deepcopy(original_organized_hyperparams_dict)
            dict_to_add = pd.read_pickle(
                '/media/deep/DATA/PycharmProjects/Summer2024/FF_editing_transfer/E_Tugolukov/Global_Part2_ML_not_DL/Regression/Results/IntermediateData/012_SVR/rbf_svr/Part2_4/Initial_NotExpanded/dict_best_parameters_values_per_fold.pkl'
                )
            logger.info(
                f"There are {len(organized_hyperparams_dict[model].keys())} in organized_hyperparams_dict with the GridSearch (sh2)")
            new_key = f"fold_{fold_idx}"
            svr_hp__1 = dict_to_add[model][new_key]
            print("only fold specific_svr_hp__1", svr_hp__1)
            first_key = list(dict_to_add[model][new_key].keys())[0]
            print("first_key", first_key)
            logger.info(f"before update {organized_hyperparams_dict[model].keys()}")
            # svr_hp__1_dic = {first_key: svr_hp__1}
            svr_hp__1 = {first_key: dict_to_add[model][new_key][first_key]}
            # print("svr_hp__1_dic",svr_hp__1_dic)
            print("svr_hp__1", svr_hp__1)
            organized_hyperparams_dict[model].update(svr_hp__1)
            full_dict[model].update(svr_hp__1)
            # organized_hyperparams_dict[model].update(svr_hp__1_dic)
            logger.info(f"after update {organized_hyperparams_dict[model].keys()}")
            logger.info(f"for fold after update fold_idx {organized_hyperparams_dict[model]}")
            logger.info(f"Checkpoint:Part4_of_5")
            logger.info(
                f"Sending in grid search parameters (doing grid search) dict for model: {model} for index: {fold_idx}")
            print("final updated dict with idx", fold_idx,
                  organized_hyperparams_dict)  # I am addin all all the best params per each fold to the overall dict
            organized_hyperparams_dict["svr_kernel_rbf"].pop("svr_kernel_rbf_hp_0", None)
            print("final updated dict with idx", fold_idx,
                  organized_hyperparams_dict)  # I am addin all all the best params per each fold to the overall dict
        elif dict_assist == "Part5_of_5":
            organized_hyperparams_dict = edit_dict_best_parameters_per_fold(original_organized_hyperparams_dict,
                                                                            fold_idx, model)
            logger.info(
                f"Sending in pre-selected grid-search parameters (post grid-search) dict for model: {model} for index: {fold_idx}")

        print(best_scaler_var[0])
        logger.info("Processing with the NOT scaling the data")

        train_X_cudf = cudf.DataFrame(dict_tmp[var1][fold_idx]['scaled_train_X'])
        train_y_cudf = cudf.Series(dict_tmp[var1][fold_idx]['unscaled_train_y'])
        val_X_cudf = cudf.DataFrame(dict_tmp[var1][fold_idx]['scaled_val_X'])
        val_y_cudf = cudf.Series(dict_tmp[var2][fold_idx]['unscaled_val_y'])

        print(train_X_cudf.shape, train_y_cudf.shape)
        print(val_X_cudf.shape, val_y_cudf.shape)

        if dict_assist == "Part4_of_5" and grid_search_status == "Grid-Search":
            print("Orientation is Part4_of_5 and Grid-Search", organized_hyperparams_dict[model])
            # llllllllllllllllllll
        if dict_assist == "Part5_of_5":
            print("Orientation is Part5_of_5 and Grid-Search", organized_hyperparams_dict[model])

        # print("try_here232323", organized_hyperparams_dict[model])

        for params_key, params in organized_hyperparams_dict[model].items():
            # print("params_key are here with dict_assist", dict_assist,params_key)
            stats = process_fold_new_svr(
                model_dict,
                logger,
                dict_assist,
                number_chosen_bins,
                grid_search_status,
                var2_class,
                var1_weight_status,
                output_out_dir,
                index=fold_idx,
                model_name=model,
                scaled_train_cudf=train_X_cudf,
                y_train_cudf=train_y_cudf,
                scaled_val_cudf=val_X_cudf,
                y_val_cudf=val_y_cudf,
                params=params,
                params_key=params_key
            )
            list_stats.append(stats)

        summary_df = pd.concat(list_stats)
    return summary_df, full_dict


def process_fold_new_svr(
        model_dict,
        logger,
        dict_assist,
        number_chosen_bins,
        grid_search_status,
        var2_class,
        var1_weight_status,
        output_dir_files,
        index,
        model_name,
        scaled_train_cudf,
        y_train_cudf,
        scaled_val_cudf,
        y_val_cudf,
        #  best_scaler, #no scaler for random forest is needed
        params,
        params_key=None):
    """
    Processes a single fold for a given model, computes metrics, and returns results as a DataFrame.
    """
    logger.info(f"Global Orientation: Processing fold {index} for model: {model_name} with params: {params_key}...")

    current_model = model_dict[model_name](**params)

    try:
        df_test_with_weights = calculate_weights_mse(y_val_cudf, number_chosen_bins)
        df_train_with_weights = calculate_weights_mse(y_train_cudf, number_chosen_bins)
        current_model.fit(scaled_train_cudf, y_train_cudf)
        logger.info(f"Model info: dict_assist is  {dict_assist} and params_key is : {params_key}")
        logger.info(
            f"Model Parameters for the fitted model: {model_name} for index: {index} for Grid-Search-Status: {grid_search_status} are: {current_model.get_params()}")

        if dict_assist == "Part5_of_5":
            model_filename = f"{output_dir_files}/model_{model_name}_fold_{index}_{var2_class}_{var1_weight_status}.pkl"
            joblib.dump(current_model, model_filename)
            logger.info(f"fitted model: {model_name} for index: {index} with best params is saved")

        predictions_train = current_model.predict(scaled_train_cudf)
        predictions_val = current_model.predict(scaled_val_cudf)
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
        val_r2 = r2_score(y_val_cudf.to_numpy(dtype=np.float64), predictions_val.to_numpy(dtype=np.float64))

        OVERFLOW_THRESHOLD = 1e150

        # Check for extreme values before computing MSE
        max_y_true = np.max(np.abs(y_val_cudf.to_numpy()))
        max_y_pred = np.max(np.abs(predictions_val))

        if max_y_pred > OVERFLOW_THRESHOLD or np.isnan(predictions_val).any() or np.isinf(predictions_val).any():
            raise ValueError(
                f"Overflow detected: Max y_pred={max_y_pred}, Min y_pred={np.min(predictions_val)}. Skipping this model run.")

    except Exception as e:
        print(f"Error processing model {model_name} with params {params_key}: {e}")
        logging.error(f"Error processing model {model_name} with params {params_key}: {e}")
        val_wmse, val_wr2 = float('nan'), float('nan')
        val_mse, val_r2 = float('nan'), float('nan')

    final_results = {
        "Model": [model_name],
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
        "Dict_Assist": [dict_assist],
    }

    return pd.DataFrame(final_results)
