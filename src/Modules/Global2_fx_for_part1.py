# Imports: GroupB: specific library: sklearn
# Imports: GroupC: specific library: cuml
# import cudf as cudf
import cupy as cp
import pandas as pd

cp.cuda.Device().use()


############################################################Data Processing fx#############################################################

def read_and_downsample_train_folds(input_dir_loc):
    """
    This fx takes the training folds and returns a full fold, and 2 downsampled versions
    """

    return_dict = {}

    # Form A, or entire df
    train_fold = pd.read_pickle(input_dir_loc)
    train_foldA = train_fold.copy(deep=True)

    # Form B is a downsampled dataset
    train_fold = pd.read_pickle(input_dir_loc)
    rows_to_add = train_fold.query("LinScore < 0.1").sample(train_fold.query("LinScore > 0.9").shape[0])
    train_df_to_be_added_too = train_fold.query("LinScore >= 0.1")
    train1 = pd.concat([train_df_to_be_added_too, rows_to_add])
    train_foldB = train1.copy(deep=True)

    return train_foldA, train_foldB
