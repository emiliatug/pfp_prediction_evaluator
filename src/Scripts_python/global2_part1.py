import os
import shutil
import sys
from typing import List  # this module allows to check the type of inputs in the terminal's py run

from Modules.Global2_fx_for_part1 import read_and_downsample_train_folds
from Utils.utils import generate_logging_file


def mainprocess(
        input1: List[str],
        input2: List[str],
        input3: List[str],
        input4: List[str],
        input5: List[str],
        org_var: List[str],
) -> None:
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)

    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info("Starting the Global Part2: Part1: Classes A and B (B is a downsampled class) generation")

    output_dir_files_train = f"{training_folds[5]}"
    os.makedirs(output_dir_files_train, exist_ok=True)

    output_dir_files_train_val = f"{training_val_folds[5]}"
    os.makedirs(output_dir_files_train_val, exist_ok=True)

    output_dir_files_val = f"{val_folds[5]}"
    os.makedirs(output_dir_files_val, exist_ok=True)

    output_dir_files_test = f"{test_folds[5]}"
    os.makedirs(output_dir_files_test, exist_ok=True)

    # Step1: Read the 5 different folds and return 2 copies: A (not downsampled), B( downsampled)
    input_dir = "../part1_data_generation_fv_02012025/train_dir/JN_gen_final/part7, where the loc variables are written from"
    for down_sampling_fold, count_fold in zip(training_folds, input5):
        logger.info(f"{count_fold} coming from: {down_sampling_fold}")
        logger.info(f"{count_fold} going to: {output_dir_files_train}{count_fold}")
        fold_train_df_A, fold_train_df_B = read_and_downsample_train_folds(down_sampling_fold)
        fold_train_df_A.to_pickle(f"{output_dir_files_train}/{count_fold}_final_train_A_df.pkl", protocol=4)
        fold_train_df_B.to_pickle(f"{output_dir_files_train}/{count_fold}_final_train_B_df.pkl", protocol=4)

    for down_sampling_fold, count_fold in zip(training_val_folds, input5):
        logger.info(f"{count_fold} coming from: {down_sampling_fold}")
        logger.info(f"{count_fold} going to: {output_dir_files_train_val}{count_fold}")
        fold_train_df_A, fold_train_df_B = read_and_downsample_train_folds(down_sampling_fold)
        fold_train_df_A.to_pickle(f"{output_dir_files_train_val}/{count_fold}_final_train_val_A_df.pkl", protocol=4)
        fold_train_df_B.to_pickle(f"{output_dir_files_train_val}/{count_fold}_final_train_val_B_df.pkl", protocol=4)
    logger.info("Step1: Making full (A) and downsampled (B) version is done")

    # Step2: Just copying hte files from the part1 to the part2's location
    for source_file, count_fold in zip(val_folds, input5):
        destination_file = f"{output_dir_files_val}{count_fold}_final_val_df.pkl"
        logger.info(f" Moving validation files for {count_fold} source_file from: {source_file}")
        logger.info(f" Moving validation files for {count_fold} destination_dir to: {destination_file}")
        shutil.copy2(source_file, destination_file)

    for source_file, count_fold in zip(test_folds, input5):
        destination_file = f"{output_dir_files_test}{count_fold}_final_test_df.pkl"
        logger.info(f" Moving test files for {count_fold} source_file from: {source_file}")
        logger.info(f" Moving test files for {count_fold} destination_dir to: {destination_file}")
        shutil.copy2(source_file, destination_file)
    logger.info("Step2: Copying Val and Test from 'Global_Part1_Data_Generation' is done")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python script.py <training_folds> , <training_val_folds>, <val_folds>, <test_folds>,<input5>,<org_var>")
        sys.exit(1)
    input1_arg = sys.argv[1]  # Collect all arguments after the script name. Those arguments are the original 4 df
    training_folds = [col.strip() for col in input1_arg.split(",")]

    input2_arg = sys.argv[2]  # Collect all arguments after the script name. Those arguments are the original 4 df
    training_val_folds = [col.strip() for col in input2_arg.split(",")]

    input3_arg = sys.argv[3]  # Collect all arguments after the script name. Those arguments are the original 4 df
    val_folds = [col.strip() for col in input3_arg.split(",")]

    input4_arg = sys.argv[4]  # Collect all arguments after the script name. Those arguments are the original 4 df
    test_folds = [col.strip() for col in input4_arg.split(",")]

    input5_arg = sys.argv[5]  # Collect all arguments after the script name. Those arguments are the original 4 df
    input5 = [col.strip() for col in input5_arg.split(",")]

    input6_arg = sys.argv[6]
    org_var = [col.strip() for col in input6_arg.split(",")]

    mainprocess(training_folds, training_val_folds, val_folds, test_folds, input5, org_var)
