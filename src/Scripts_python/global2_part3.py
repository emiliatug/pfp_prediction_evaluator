# imports are splits by the groups:
# Imports: GroupA: general
# Imports: GroupD: general: system, and GPU related
import os
import sys
import time

import cudf as cudf
import cupy as cp
import pandas as pd

cp.cuda.Device().use()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Now import modules
from Modules.Global2_fx_for_part4_rf import (
    calculate_weights,
    preprocess_data_for_gpu1,
    choosing_best_scaler_selection1,
    preprocess_train_and_val_data_for_scalers
)

from Modules.Global2_fx_for_linear_models_script import generate_stats_df_of_best_scaler

from Utils.utils import generate_logging_file


def choosing_best_scaler_generation1(model_list, input_loc_train, ordinal_features_not_to_weight,
                                     columns_not_to_scale, var1_weight_status, var2_class, scaler_list):
    """This fx combines the fx above: it first calls, 'calculate_weights' to get the weight for the train_input df, and then it calls,
        'preprocess_train_and_val_data_with_weight_feature' in oerder to get 6 df, and then it calls, 'generate_stats_df_of_best_scaler' to 
        generate a stats for the 1 CV across all scalers across all models
        """

    tmp_df_train = pd.read_pickle(input_loc_train)
    train_fold_df_post_norm = calculate_weights(tmp_df_train)

    input_train_X, input_train_y, X_weighted, y_weighted = preprocess_train_and_val_data_for_scalers(
        train_fold_df_post_norm,
        ordinal_features_not_to_weight,
    )

    input_train_X = preprocess_data_for_gpu1(input_train_X, columns_not_to_scale)
    X_weighted = preprocess_data_for_gpu1(X_weighted, columns_not_to_scale)
    input_train_y = preprocess_data_for_gpu1(input_train_y, columns_not_to_scale)
    y_weighted = preprocess_data_for_gpu1(y_weighted, columns_not_to_scale)

    # fx **generate_stats_df_of_best_scaler** is being used inside **choosing_best_scaler_generation1**
    start_time = time.time()
    result_total_df = generate_stats_df_of_best_scaler(model_list, input_train_X, X_weighted, input_train_y, y_weighted,
                                                       columns_not_to_scale, var1_weight_status,
                                                       var2_class, scaler_list)
    end_time = time.time()
    time_difference = end_time - start_time
    return result_total_df


def mainprocess(
        columns_not_to_scale: list[str],
        ordinal_features_not_to_weight: list[str],
        model_list: list[str],
        input_loc_train: list[str],
        var1_weight_status: list[str],
        var2_class: str,
        org_var: list[str],
        # logging_file_name:str,
        # output_dir_files: str,
        scaler_list: list[str],

) -> None:
    # Set up logging files
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    # Part0.5 Set up logger file
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info("Starting the Global Part2: Part3: Assessing Reasonable Scalers")

    # Part1: There are 3 fx that I want to call together, and this fx will be called as a unit multiple times, but I still want to see it

    for model in model_list:
        # print("start", model)
        output_dir_files = os.path.join(org_var[2], model, org_var[3])
        # print("files", output_dir_files)
        os.makedirs(output_dir_files, exist_ok=True)
        list_total_L, list_best_L = [], []
        for index, fold_and_dataset in enumerate(input_loc_train[0:5]):
            print(index, fold_and_dataset)
            letter = fold_and_dataset[-8:-7]
            fold_and_dataset_var = fold_and_dataset[2:-7]
            var2_class_mapping = "Bal" + letter
            for j, weight in enumerate(var1_weight_status):
                logger.info(
                    f"The scaling is being tested on: model x fold_and_dataset(A or B) x weight var: {model}_____X_____{fold_and_dataset_var}_____X_____{weight}")

                result_total_df = choosing_best_scaler_generation1(
                    model, input_loc_train[index],
                    ordinal_features_not_to_weight, columns_not_to_scale,
                    var1_weight_status[j], var2_class_mapping, scaler_list)
                result_total_post_selection, best_scaler_post_selection = choosing_best_scaler_selection1(
                    result_total_df, var1_weight_status[j], var2_class_mapping)
                result_total_post_selection["Fold"] = "Fold:" + str(index)
                best_scaler_post_selection["Fold"] = "Fold:" + str(index)
                list_total_L.append(result_total_post_selection)
                list_best_L.append(best_scaler_post_selection)
        print(len(list_best_L))

        total_df1 = pd.concat(list_total_L, axis=0)
        best_df1 = pd.concat(list_best_L, axis=0)
        print(total_df1.shape, best_df1.shape)
        total_df1.to_pickle(f'{output_dir_files}/{model_list[0]}_total_df_BalB.pkl', protocol=4)
        best_df1.to_pickle(f'{output_dir_files}/{model_list[0]}_best_df_BalB.pkl', protocol=4)
        logger.info(
            f"The model {model} has been scaled and saved for all scaler inputs, both weighted and unweighted, for BalB.")
    logger.info("Step3 is done")


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(
            "Usage: python Part2_4.py <columns_not_to_scale>, <ordinal_features_not_to_weight>, <model_list>, <input_loc_train>, <var1_weight_status>, <var2_class>, <org_var>,<scaler_list>")
        sys.exit(1)
    columns_not_to_scale = sys.argv[1]
    ordinal_features_not_to_weight = sys.argv[2]
    model_list = sys.argv[3]
    input_loc_train = sys.argv[4]
    var1_weight_status = sys.argv[5]
    var2_class = sys.argv[6]
    org_var = sys.argv[7]
    scaler_list = sys.argv[8]
    org_var = [col.strip() for col in org_var.split(",")]
    input_loc_train = [col.strip() for col in input_loc_train.split(",")]
    var1_weight_status = [col.strip() for col in var1_weight_status.split(",")]
    var2_class = [col.strip() for col in var2_class.split(",")]
    columns_not_to_scale = [col.strip() for col in columns_not_to_scale.split(",")]
    ordinal_features_not_to_weight = [col.strip() for col in ordinal_features_not_to_weight.split(",")]
    model_list = [col.strip() for col in model_list.split(",")]
    scaler_list = [col.strip() for col in scaler_list.split(",")]

    mainprocess(columns_not_to_scale, ordinal_features_not_to_weight, model_list, input_loc_train, var1_weight_status,
                var2_class, org_var, scaler_list)
