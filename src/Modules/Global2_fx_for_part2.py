# Imports: GroupC: specific library: cuml
import cudf as cudf
import cupy as cp
import numpy as np
import pandas as pd
# Imports: GroupB: specific library: sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

cp.cuda.Device().use()


############################################################Generating Naive Models#####################################################


def make_naive_models(y_train):
    """ 
    This fx will make a naive model by just taking the mean of the labels
    """

    length_oflists = len(y_train)
    naiveMean = np.mean(y_train)
    return naiveMean


def print_stats_naive(input1, input2, input3):
    """
    This fx will generate the mse,r2 and the mae for the naive model
    """

    input1 = np.full(len(input2), input1)
    mse = mean_squared_error(input2, input1)
    r2 = r2_score(input2, input1)
    mae = mean_absolute_error(input2, input1)
    return input3, mse, r2, mae


def generate_naive_mean_models(input_train_dir, input_test_dir, names_naive_models, output_dir):
    """
    This fx generates very naive model where the prediction label is the average of the labels in the train dataset
    """

    # part1 generate naive model
    results = []
    train_df = pd.read_pickle(f'{input_train_dir}')
    naiveMean = pd.Series(make_naive_models(train_df["LinScore"]))
    naiveMean.to_pickle(f'{output_dir}/{names_naive_models}.pkl')
    # part2: read the test data and compare it to the naive model
    test_df = pd.read_pickle(f'{input_test_dir}')
    results.append(print_stats_naive(naiveMean, train_df["LinScore"], "Train_Comparison"))
    results.append(print_stats_naive(naiveMean, test_df["LinScore"], "Test_Comparison"))
    results_df = pd.DataFrame(results, columns=["Type", "MSE", "R2", "MAE"])
    results_df["State:A,B"] = names_naive_models
    return results_df
