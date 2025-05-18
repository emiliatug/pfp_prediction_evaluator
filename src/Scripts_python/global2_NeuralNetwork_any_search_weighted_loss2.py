import json
# Imports: GroupD: general: system, and GPU related
import os
import sys
# rmm.reinitialize(pool_allocator=True, initial_pool_size=18 * 1024 ** 3)  # Use up to 16 GB of GPU memory
from typing import List, Dict

# Imports: GroupC: specific library: cuml
import cudf as cudf
import cupy as cp
import rmm  # RAPIDS Memory Manager
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
# Imports: GroupB: specific library: sklearn
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, MaxAbsScaler, \
    QuantileTransformer, FunctionTransformer
from tensorflow import keras

cp.cuda.Device().use()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.utils import generate_logging_file
from cuml.svm import LinearSVR
from cuml.svm import SVR
from cuml.internals.logger import set_level, level_trace
from Modules.Global2_fx_for_linear_models_script import (cv_with_gridserch_for_linear_fx,
                                                         read_reweight_and_preprocess_folds, preprocess_data_for_gpu3,
                                                         preprocess_data_for_gpu3_5, calculate_weights_mse)
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad

os.environ[
    "TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Enable cuda_malloc_async for asynchronous memory allocation (fixes fragmented memory)


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
        nodes_per_layer_configurations_1to9: Dict[str, List],
        optimizers_list: List[str],
        input15: List[str],
) -> None:
    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # set the output dir for the logging file
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    use_sample_weights = org_var[3] == "weighted_loss"
    saved_models_loc = org_var[5]

    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)
    logger.info(f"Starting the Global Part2: Training Best NN per fold, and saving their stats")

    for model_name in model_list:

        # make **output_dir_files**, where all the files will be stored
        output_dir_files = org_var[2]
        os.makedirs(output_dir_files, exist_ok=True)
        output_dir = org_var[2]
        os.makedirs(output_dir, exist_ok=True)
        print(model_name, output_dir_files, output_dir)
        # output_dir_files_models = os.path.join(output_dir_files, "models")
        # os.makedirs(output_dir_files_models, exist_ok=True)

        # get the variables names
        #     grid_search_status=org_var[3] ###NOT NEEDED
        num_folds = num_var[0]
        number_chosen_bins = num_var[1]
        model_dict = {"neural_network_ning": "neural_network_manual_tuning"}
        scaler_dict = {
            "MaxAbsScaler": MaxAbsScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
            "PowerTransformer": PowerTransformer(method="yeo-johnson", standardize=True),
            "StandardScaler": StandardScaler(),
            "LogScaler": FunctionTransformer(np.log1p, validate=True)
        }

        # assigning optimizers to optimizers_dict:
        list_optimizers_fx = [SGD, Adam, RMSprop, Adagrad]
        list_optimizers_fx_names = ["SGD", "Adam", "RMSprop", "Adagrad"]
        optimizers = dict(zip(list_optimizers_fx_names, list_optimizers_fx))
        optimizer_dict = {}
        for name in optimizers_list:
            optimizer_dict[name] = optimizers[name]
        optimizers = optimizer_dict

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

        CV0_test_df = pd.read_pickle(input12[0])
        CV1_test_df = pd.read_pickle(input12[1])
        CV2_test_df = pd.read_pickle(input12[2])
        CV3_test_df = pd.read_pickle(input12[3])
        CV4_test_df = pd.read_pickle(input12[4])

        logger.info(f"Shape of the **train_df** is {CV0_train_df.shape} and **val_df** {CV0_train_df.shape}")
        logger.info("All the variables and loaded")
        logger.info("The approximate shape of train data is 400k. Is above shape 400K?")

        # Now that all the variables are loaded, the code is starting
        # Step1: generate dict_cv_data,best_scaler_per_model_df: I left the weight parameter in, but in most cases (all except OLS), it made it 'unweighted'. In case I had to go back
        dict_folds_with_weights = read_reweight_and_preprocess_folds(
            # Test Data (List of 5 DataFrames, one per fold)
            [
                CV0_test_df,  # test DataFrame for Fold 0
                CV1_test_df,  # test DataFrame for Fold 1
                CV2_test_df,  # test DataFrame for Fold 2
                CV3_test_df,  # test DataFrame for Fold 3
                CV4_test_df  # test DataFrame for Fold 4
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
            dict_folds_with_weights,  # Dictionary of training/test folds with sample weights
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

        print(type(dict_cv_data_cpu))
        print(dict_cv_data_cpu.keys())
        print(dict_cv_data_cpu['neural_network_manual_tuning'].keys())
        print(dict_cv_data_cpu['neural_network_manual_tuning'][0].keys())

        list_dicts_to_collect_global_final_results = []
        if org_var[4] == "very_big_search":
            print("correct search type")
            fold_list = [0]
            hidden_layers_list = ["hidden_layers1", "hidden_layers2", "hidden_layers3", "hidden_layers4",
                                  "hidden_layers5",
                                  "hidden_layers6", "hidden_layers7", "hidden_layers8", "hidden_layers9",
                                  "hidden_layers10",
                                  "hidden_layers20", "hidden_layers30", "hidden_layers40", "hidden_layers50"
                                  ]
            opt_name_list = ["Adagrad", "Adam", "RMSprop", "SGD"]
        elif org_var[4] == "big_search":
            # print("input15",input15)
            # print("optimizers_list",optimizers_list)
            fold_list = [0, 1, 2, 3, 4]
            hidden_layers_list = [f"hidden_layers{int(i)}" for i in input15]
            # hidden_layers_list=["hidden_layers1","hidden_layers5","hidden_layers10","hidden_layers15","hidden_layers20"]
            # opt_name_list=["Adagrad", "Adam","RMSprop"]
        elif org_var[4] == "small_search":
            fold_list = [0, 1, 2, 3, 4]
            hidden_layers_list = [f"hidden_layers{int(i)}" for i in input15]
            # hidden_layers_list=["hidden_layers1","hidden_layers2","hidden_layers3","hidden_layers4","hidden_layers5",
            #                     "hidden_layers6","hidden_layers7","hidden_layers8","hidden_layers9"]
            # opt_name_list=["Adagrad", "Adam","RMSprop"]
        elif org_var[4] == "epoch_opt_no_early_stop":
            fold_list = [0, 1, 2, 3, 4]
            hidden_layers_list = [f"hidden_layers{int(i)}" for i in input15]

        list_results = []
        dict_best_model_per_fold = {}
        for fold in fold_list:
            X_train, y_train, X_test, y_test = (
                dict_cv_data_cpu[model_name][fold]['scaled_train_X'],
                dict_cv_data_cpu[model_name][fold]['unscaled_train_y'],
                dict_cv_data_cpu[model_name][fold]['scaled_val_X'],
                dict_cv_data_cpu[model_name][fold]['unscaled_val_y'])
            df_with_weights_mse_test = calculate_weights_mse(cudf.Series(y_test), 101)

            for hidden_layers in hidden_layers_list:
                # for opt_name in opt_name_list:
                for opt_name in optimizers_list:
                    model_loc_keras = f"{saved_models_loc}/NN_model_manually_tunned_{org_var[3]}_fold{fold}_{hidden_layers}_{opt_name}.keras"
                    model_loc_h5 = f"{saved_models_loc}/NN_model_manually_tunned_{org_var[3]}_fold{fold}_{hidden_layers}_{opt_name}.h5"

                    if os.path.exists(model_loc_keras):
                        model_path = model_loc_keras
                        print("model_loc path is", model_loc_keras)
                        model = keras.models.load_model(model_path)
                        # model_name
                    elif os.path.exists(model_loc_h5):
                        model_path = model_loc_h5
                        print("model_loc path is", model_loc_h5)
                        model = keras.models.load_model(model_path)

                    predictions = model.predict(X_test)
                    print("model is loaded")
                    fold_and_model_specific_mse_test = sk_mse(y_test, predictions)
                    fold_and_model_specific_wmse = sk_mse(y_test, predictions,
                                                          sample_weight=df_with_weights_mse_test["Weight"])
                    fold_and_model_specific_r2 = sk_r2(y_test, predictions)
                    fold_and_model_specific_wr2 = sk_r2(y_test, predictions,
                                                        sample_weight=df_with_weights_mse_test["Weight"])
                    predictions = [item for sublist in predictions for item in sublist]
                    model_path_name = model_path.split("/")[-1].strip()
                    # model_path_name=model_path_name[:-3]
                    model_path_name = model_path_name.replace("hidden_layers", "hl")
                    # model_path_name=model_path_name.replace("manually_tunned","man").strip()
                    list_results.append([model_path_name, fold, hidden_layers, opt_name, fold_and_model_specific_wmse,
                                         fold_and_model_specific_wr2, fold_and_model_specific_mse_test,
                                         fold_and_model_specific_r2, model, y_test.to_list(), predictions])

                    # save the best model according to the wr2
                    dict_best_model_per_fold[fold] = model
                    # make it into a dic, then turn that dict into a df line then add it to the list, then concat
                    dicts_to_collect_global_final_results = {}
                    # print("dict1",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Tuning Type"] = org_var[4]
                    # print("dict2",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Loss Type"] = org_var[3]
                    # print("dict3",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Fold"] = fold
                    # print("dict4",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Hidden layers"] = hidden_layers
                    # print("dict5",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Opt name"] = opt_name
                    # print("dict6",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["Test mse per model"] = fold_and_model_specific_mse_test
                    # print("dict7",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["W. Test mse per model"] = fold_and_model_specific_wmse
                    # print("dict8",dicts_to_collect_global_f inal_results)
                    dicts_to_collect_global_final_results["Test R2 per model"] = fold_and_model_specific_r2
                    # print("dict9",dicts_to_collect_global_final_results)
                    dicts_to_collect_global_final_results["W. Test R2 per model"] = fold_and_model_specific_wr2
                    # print("dict10",dicts_to_collect_global_final_results)
                    list_dicts_to_collect_global_final_results.append(
                        pd.DataFrame([dicts_to_collect_global_final_results]))
                    # print("list",list_dicts_to_collect_global_final_results)

        df_results = pd.DataFrame(list_results,
                                  columns=["Model_Path", "Fold", "Hidden_Layers", "Optimizer", "Weighted Test MSE",
                                           "Weighted Test R²", "Test MSE", "Test R²", "Model", "Y_test", "Predictions"])

        # Get the row with the maximum fold_and_model_specific_wr2
        best_models_per_fold_stats = df_results.loc[
            df_results.groupby("Fold")["Weighted Test R²"].idxmax()].reset_index(drop=True)

        # save the best model
        for index, row in best_models_per_fold_stats.iterrows():
            model = row["Model"]
            fold = row["Fold"]
            hidden_layers = row["Hidden_Layers"]
            opt_name = row["Optimizer"]
            model.save(f'{output_dir_files}/model_NN_{org_var[3]}_fold{fold}_{hidden_layers}_{opt_name}.keras')

        print(best_models_per_fold_stats.shape)
        print(best_models_per_fold_stats)
        real_list = best_models_per_fold_stats["Y_test"].to_list()
        predictions_list = best_models_per_fold_stats["Predictions"].to_list()
        print("length of the individual y_real for each fold", len(real_list[0]), len(real_list[1]), len(real_list[2]),
              len(real_list[3]), len(real_list[4]))

        real_list = [item for sublist in best_models_per_fold_stats["Y_test"].tolist() for item in sublist]
        predictions_list = [item for sublist in best_models_per_fold_stats["Predictions"].tolist() for item in sublist]

        df_with_weights_mse_test = calculate_weights_mse(cudf.Series(real_list), 101)
        print("length of the combined y_real and y_predicted for all folds", len(real_list), len(predictions_list))
        folds_combined_mse_test = sk_mse(real_list, predictions_list)
        folds_combined_r2_test = sk_r2(real_list, predictions_list)
        folds_combined_wmse_test = sk_mse(real_list, predictions_list, sample_weight=df_with_weights_mse_test["Weight"])
        folds_combined_wr2_test = sk_r2(real_list, predictions_list, sample_weight=df_with_weights_mse_test["Weight"])
        print(best_models_per_fold_stats.shape)
        best_models_per_fold_stats["total_mse"] = folds_combined_mse_test
        best_models_per_fold_stats["total_r2"] = folds_combined_r2_test
        best_models_per_fold_stats["total_wmse"] = folds_combined_wmse_test
        best_models_per_fold_stats["total_wr2"] = folds_combined_wr2_test
        print(best_models_per_fold_stats.shape)
        print(best_models_per_fold_stats)
        best_models_per_fold_stats = best_models_per_fold_stats.rename(columns={
            "Hidden_Layers": "Hidden layers",
            "Optimizer": "Opt name"
        })

    tmp_df = pd.concat(list_dicts_to_collect_global_final_results, axis=0).reset_index(drop=True)
    combined_df = tmp_df.merge(best_models_per_fold_stats, how="left", on=["Fold", "Hidden layers", "Opt name"])
    best_models_per_fold_stats = best_models_per_fold_stats.drop(["Model"], axis=1)
    best_models_per_fold_stats = best_models_per_fold_stats.rename(columns={"Model_Path": "Model"})
    best_models_per_fold_stats["Model"] = best_models_per_fold_stats["Model"].str.strip()
    best_models_per_fold_stats.to_pickle(f'{output_dir_files}/best_models_per_fold_stats_{org_var[4]}_{org_var[3]}.pkl')
    combined_df.to_pickle(f'{output_dir_files}/Final_Test_Results_NN{org_var[4]}_{org_var[3]}.pkl')


if __name__ == "__main__":
    if len(sys.argv) != 16:
        print(
            "Usage: python Part2_5.py <Columns_not_to_scale> <Ordinal_features_not_to_weight> <input3_arg List> <input4_arg> <num_folds>, <best_scaler_var>,<var1_weight_status>,<var2_class>,<model_list>,<org_var>,<nodes_per_layer_configurations_1to9>,<optimizers_dict>,<input15")
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
    nodes_per_layer_configurations_1to9 = json.loads(sys.argv[13])
    optimizers_list = sys.argv[14]
    input15_arg = sys.argv[15]

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
    input15 = [col.strip() for col in input15_arg.split(",")]
    optimizers_list = [col.strip() for col in optimizers_list.split(",")]
    mainprocess(columns_not_to_scale, ordinal_features_not_to_weight, input3, input4, num_var, best_scaler_var
                , var1_weight_status, var2_class, model_list, org_var, input11, input12,
                nodes_per_layer_configurations_1to9, optimizers_list, input15)
