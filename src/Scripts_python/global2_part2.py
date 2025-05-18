# Imports: GroupB: specific library: sklearn
# #Imports: GroupC: specific library: cuml
# import cudf as cudf
# from cuml.linear_model import Lasso
# import cuml
# from cupy import asnumpy
# import cupy as cp
# from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
# from cuml import LinearRegression
# from cuml.linear_model import LinearRegression
# Imports: GroupD: general: system, and GPU related
import os
import sys
from collections import namedtuple
from typing import List

import pandas as pd

# sys.path.append("../part1_data_generation_fv_02012025")
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Now import modules

from Modules.Global2_fx_for_part2 import generate_naive_mean_models
from Utils.utils import generate_logging_file


def mainprocess(
        input_train_dir: List[str],
        input_test_dir: str,
        names_naive_models: List[str],
        org_var: List[str],
) -> None:
    output_dir = org_var[1]
    os.makedirs(output_dir, exist_ok=True)

    # Part0.5 Set up logger file
    logger = generate_logging_file(name=f'{org_var[2]}', loc_to_write=output_dir)
    logger.info("Starting the Global Part2: Part2: Naive Model Generation")

    output_dir_files = org_var[0]
    os.makedirs(output_dir_files, exist_ok=True)
    # # Define the named tuple
    Args = namedtuple('Args', ["input_loc", "test_loc", "names_of_model", "output_dir_files"])

    config = []
    for i in range(len(input_train_dir)):
        config.append(Args(input_train_dir[i], input_test_dir[i], names_naive_models[i % 2], output_dir_files))

    results = []  # Initialize results list
    for args in config:
        logger.info(args)
        df = generate_naive_mean_models(*args)  # Unpack namedtuple into function arguments
        results.append(df)
    stats_training_test_df = pd.concat(results, ignore_index=True)
    stats_training_test_df["Fold"] = stats_training_test_df.index // 4
    stats_training_test_df.to_pickle(f'./{output_dir_files}/stats_df_combined.pkl', protocol=4)
    logger.info("Step1 is done")
    logger.info("Part2_2.py has finished running")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python Part2_4.py <input_train_dir> <input_test_dir/B> <naive_model_name> <org_var >")
        sys.exit(1)
    input1_arg = sys.argv[1]
    input_train_dir = [col.strip() for col in input1_arg.split(",")]
    input2_arg = sys.argv[2]
    input_test_dir = [col.strip() for col in input2_arg.split(",")]
    input3_arg = sys.argv[3]
    names_naive_models = [col.strip() for col in input3_arg.split(",")]
    input4_arg = sys.argv[4]
    org_var = [col.strip() for col in input4_arg.split(",")]

mainprocess(input_train_dir, input_test_dir, names_naive_models, org_var)
