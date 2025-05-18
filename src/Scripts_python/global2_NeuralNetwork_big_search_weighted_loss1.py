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
from Modules.Global2_fx_for_neural_networks import build_nn_model
import pandas as pd
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adagrad
import pickle
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import get as get_loss  # Add this import if not already present


class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, org_var, fold, n_hidden_layers, opt_name,
                 X_train, y_train, sample_weight=None, sample_size=1024, logger=None):
        super().__init__()
        self.log_dir = log_dir
        self.org_var = org_var
        self.fold = fold
        self.n_hidden_layers = n_hidden_layers
        self.opt_name = opt_name
        self.global_step = 0
        self.logger = logger

        idx = np.random.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
        self.X_sample = tf.convert_to_tensor(X_train.iloc[idx], dtype=tf.float32)
        self.y_sample = tf.convert_to_tensor(y_train.iloc[idx], dtype=tf.float32)
        self.sw_sample = (
            tf.convert_to_tensor(sample_weight.iloc[idx], dtype=tf.float32)
            if sample_weight is not None else None
        )

        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(self.X_sample, training=True)
                loss_fn = self.model.loss
                if isinstance(loss_fn, str):
                    loss_fn = get_loss(loss_fn)

                # Manually compute weighted loss
                if self.sw_sample is not None:
                    unweighted_loss = loss_fn(self.y_sample, predictions)
                    loss_value = tf.reduce_mean(unweighted_loss * self.sw_sample)
                else:
                    loss_value = loss_fn(self.y_sample, predictions)

            gradients = tape.gradient(loss_value, self.model.trainable_weights)

            with self.writer.as_default():
                for weight, grad in zip(self.model.trainable_weights, gradients):
                    if grad is not None:
                        grad_val = grad.numpy()
                        mean_abs_grad = np.mean(np.abs(grad_val))
                        l2_norm = np.linalg.norm(grad_val)

                        weight_name = weight.name.replace(":", "_").replace("/", "_")
                        tag_prefix = f"gradients/{self.org_var}/fold{self.fold}/layers{self.n_hidden_layers}/{self.opt_name}/{weight_name}"

                        tf.summary.scalar(f"{tag_prefix}/mean_abs", mean_abs_grad, step=self.global_step)
                        tf.summary.scalar(f"{tag_prefix}/l2_norm", l2_norm, step=self.global_step)
                        tf.summary.histogram(f"{tag_prefix}/hist", grad_val, step=self.global_step)

            self.global_step += 1
            self.writer.flush()
            self.logger.info(
                f"[✅ GradientLogger] Gradients logged at step {self.global_step} for fold {self.fold}, layers={self.n_hidden_layers}, opt={self.opt_name}")

        except Exception as e:
            self.logger.info(f"[⚠️ GradientLogger] Epoch {epoch} failed to log gradients: {e}")


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
    # set the output dir for the logging file
    output_dir = org_var[0]
    os.makedirs(output_dir, exist_ok=True)
    use_sample_weights = org_var[3] == "weighted_loss"

    # Part0.5 Set up logger file     
    logger = generate_logging_file(name=f"{org_var[1]}", loc_to_write=output_dir)

    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logger.info(e)

    logger.info(f"Starting the Global Part2: Training Best NN per fold, and saving their stats")

    for model_name in model_list:

        # make **output_dir_files**, where all the files will be stored
        output_dir_files = org_var[2]
        os.makedirs(output_dir_files, exist_ok=True)
        output_dir = org_var[2]
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"{model_name}, {output_dir_files}, {output_dir}")
        output_dir_files_models = os.path.join(output_dir_files, "models")
        os.makedirs(output_dir_files_models, exist_ok=True)
        # output_dir_tensorflow_logs = os.path.join(output_dir_files, "logs")
        # os.makedirs(output_dir_tensorflow_logs, exist_ok=True)

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
        list_optimizers_fx = [Adam, Adagrad]
        list_optimizers_fx_names = ["Adam", "Adagrad"]
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

        CV0_val_df = pd.read_pickle(input4[0])
        CV1_val_df = pd.read_pickle(input4[1])
        CV2_val_df = pd.read_pickle(input4[2])
        CV3_val_df = pd.read_pickle(input4[3])
        CV4_val_df = pd.read_pickle(input4[4])

        # DELETEDELETEDELETE
        # CV0_val_df=CV0_val_df.sample(frac=0.00025, random_state=42)
        # CV1_val_df=CV1_val_df.sample(frac=0.00025, random_state=42)
        # CV2_val_df=CV2_val_df.sample(frac=0.00025, random_state=42)
        # CV3_val_df=CV3_val_df.sample(frac=0.00025, random_state=42)
        # CV4_val_df=CV4_val_df.sample(frac=0.00025, random_state=42)
        # CV0_train_df=CV0_train_df.sample(frac=0.00025, random_state=42)
        # CV1_train_df=CV1_train_df.sample(frac=0.00025, random_state=42)
        # CV2_train_df=CV2_train_df.sample(frac=0.00025, random_state=42)
        # CV3_train_df=CV3_train_df.sample(frac=0.00025, random_state=42)
        # CV4_train_df=CV4_train_df.sample(frac=0.00025, random_state=42)
        # logger.info("input is downsampled")
        # DELETEDELETEDELETE

        logger.info(f"Shape of the **train_df** is {CV0_train_df.shape} and **val_df** {CV0_val_df.shape}")
        logger.info("All the variables and loaded")
        logger.info("The approximate shape of train data is 400k. Is above shape 400K?")

        # Now that all the variables are loaded, the code is starting
        # Step1: generate dict_cv_data,best_scaler_per_model_df: I left the weight parameter in, but in most cases (all except OLS), it made it 'unweighted'. In case I had to go back
        # dict_folds_with_weights, df_train_val_with_weights, df_val_with_weights = read_reweight_and_preprocess_folds(
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
            logger.info("should not be here")
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

        logger.info(type(dict_cv_data_cpu))
        logger.info(dict_cv_data_cpu.keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'].keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'][0].keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'][1].keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'][2].keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'][3].keys())
        # logger.info(dict_cv_data_cpu['neural_network_manual_tuning'][4].keys())

        default_lr = 0.001  # learning rate
        for fold in dict_cv_data_cpu[model_name].keys():
            # if fold==3 or fold==2:

            logger.info(f"Processing Fold {fold} for NN training")
            X_train, y_train, X_val, y_val = (
                dict_cv_data_cpu[model_name][fold]['scaled_train_X'],
                dict_cv_data_cpu[model_name][fold]['unscaled_train_y'],
                dict_cv_data_cpu[model_name][fold]['scaled_val_X'],
                dict_cv_data_cpu[model_name][fold]['unscaled_val_y'])
            logger.info(f"{type(X_train)},{X_train.shape}")

            input_dim = X_train.shape[1]
            df_with_weights_mse_train = calculate_weights_mse(cudf.Series(y_train), 101)
            df_with_weights_mse_val = calculate_weights_mse(cudf.Series(y_val), 101)
            dict_history, dict_nn_global_results = {}, {}

            for n_hidden_layers in input15:
                dict_history[n_hidden_layers] = {}
                dict_nn_global_results[n_hidden_layers] = {}
                # n_hidden_layers=int(n_hidden_layers)

                for opt_name, opt in optimizers.items():

                    log_dir = os.path.join(
                        output_dir_files, "logs",
                        f"NN_model_manually_tunned_{org_var[3]}_fold{fold}_hidden_layers{n_hidden_layers}_{opt_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

                    os.makedirs(log_dir, exist_ok=True)

                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                    logger.info(
                        f"Processing Fold {fold} with {n_hidden_layers} hidden layers and 100 epochs early stopping for {opt_name} ...")
                    lr = default_lr
                    K.clear_session()
                    gc.collect()

                    model = build_nn_model(input_dim, opt, lr, n_hidden_layers, df_with_weights_mse_train,
                                           nodes_per_layer_configurations_1to9)

                    logger.info("Starting check of the layer archtecture")
                    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

                    for idx, layer in enumerate(dense_layers):
                        role = "Hidden"
                        if idx == 0:
                            role = "Input"
                        elif idx == len(dense_layers) - 1:
                            role = "Output"
                        logger.info(f"{role} Layer {idx + 1}: {layer.units} neurons")
                        logger.info("Ending check of the layer archtecture")

                    early_stopping = EarlyStopping(
                        monitor="val_loss",
                        min_delta=0.000001,
                        patience=10,
                        verbose=2,
                        restore_best_weights=True  # Restore the best weights
                    )

                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,  # changed from defualt 0.1 to 0.5
                        patience=10,  # wait 10 epochs before doing early stope
                        verbose=1,
                        mode='auto',
                        min_delta=0.0001,  # olerance parameter
                        cooldown=0,  # how many epcochs after lr is cut before the new checking can start
                        min_lr=1e-6  # if it comes to this, then scheduler will stop
                    )

                    sample_weight_train = df_with_weights_mse_train["Weight"] if use_sample_weights else None
                    sample_weight_val = df_with_weights_mse_val["Weight"] if use_sample_weights else None

                    gradient_callback = GradientLogger(
                        log_dir=log_dir,
                        org_var=org_var[3],  # assuming org_var[3] contains the "weighted_loss" or identifier string
                        fold=fold,
                        n_hidden_layers=n_hidden_layers,
                        opt_name=opt_name,
                        X_train=X_train,
                        y_train=y_train,
                        sample_weight=sample_weight_train,
                        sample_size=1024,  # feel free to change to smaller (e.g., 512) if memory is tight
                        logger=logger
                    )

                    history = model.fit(
                        X_train, y_train,
                        sample_weight=sample_weight_train,
                        validation_data=(X_val, y_val, sample_weight_val) if use_sample_weights else (X_val, y_val),
                        epochs=100,
                        batch_size=256,
                        callbacks=[early_stopping, lr_scheduler, tensorboard_callback, gradient_callback],
                        verbose=1
                    )

                    model.save(
                        f"{output_dir_files_models}/NN_model_manually_tunned_{org_var[3]}_fold{fold}_hidden_layers{n_hidden_layers}_{opt_name}.keras")

                    # write up the averaged BATCHES per epoch
                    dict_history[n_hidden_layers][opt_name] = history.history

                    # using a trained model,
                    y_val_pred = model.predict(X_val)
                    per_epoch_mse_val = sk_mse(y_val, y_val_pred)
                    per_epoch_wmse_val = sk_mse(y_val, y_val_pred, sample_weight=df_with_weights_mse_val["Weight"])
                    per_epoch_r2_val = sk_r2(y_val, y_val_pred)
                    per_epoch_wr2_val = sk_r2(y_val, y_val_pred, sample_weight=df_with_weights_mse_val["Weight"])

                    per_epoch_metrics = {
                        "mse_val": per_epoch_mse_val,
                        "wmse_val": per_epoch_wmse_val,
                        "r2_val": per_epoch_r2_val,
                        "wr2_val": per_epoch_wr2_val
                    }

                    dict_nn_global_results[n_hidden_layers][opt_name] = {}
                    dict_nn_global_results[n_hidden_layers][opt_name] = pd.DataFrame([per_epoch_metrics])

            K.clear_session()
            gc.collect()
            logger.info(type(dict_history))
            logger.info(dict_history.keys())

            with open(f'{output_dir_files}/dict_history_{org_var[4]}_{org_var[3]}_fold{fold}.pkl', 'wb') as f:
                pickle.dump(dict_history, f)
            with open(f'{output_dir_files}/dict_nn_global_results_{org_var[4]}_{org_var[3]}{fold}.pkl', 'wb') as f:
                pickle.dump(dict_nn_global_results, f)


if __name__ == "__main__":
    if len(sys.argv) != 16:
        logger.info(
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
