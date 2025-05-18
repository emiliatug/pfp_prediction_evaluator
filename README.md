# Protein Function Prediction ML Evaluation

This project implements supervised ML pipelines for regression tasks using CuML, FLAML, scikit-learn, and Keras to evaluate and improve Lin Score prediction accuracy, focusing on loss optimization and bin-wise error reduction.

## Environment

Please activate the environment before starting:

```bash
pip install requirements.txt
```

---
## Step 1: Prepare and Scale Data

### Generate Cross-Validation Datasets

```bash
chmod +x ./Scripts_bash/run_global2_part12.sh
./Scripts_bash/run_global2_part12.sh
```

### Run Feature Scalers

```bash
chmod +x ./Scripts_bash/run_global2_part3.sh
./Scripts_bash/run_global2_part3.sh
```

---

## Step 2: Prepare Logging Directories

```bash
mkdir -p Log_files/Linear/stdout_stderr
mkdir -p Log_files/Decision_Trees/stdout_stderr
mkdir -p Log_files/SVR/stdout_stderr
mkdir -p Log_files/NN/stdout_stderr
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
```

---

## Step 3: Run Models

### A. Linear Models

#### 1. Ordinary Least Squares (OLS) and Weighted Least Squares (WLS)

```bash
./Scripts_bash/run_global2_linear_ols.sh > Log_files/Linear/stdout_stderr/global2_linear_ols_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_ols_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_wls.sh > Log_files/Linear/stdout_stderr/global2_linear_wls_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_wls_stderr${TIMESTAMP}.log
```

#### 2. Mini-batch SGD

```bash
./Scripts_bash/run_global2_linear_mb_SGD_logS.sh > Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_logS_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_logS_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_mb_SGD_log_and_StandardS.sh > Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_log_and_StandardS_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_log_and_StandardS_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_mb_SGD_standardS.sh > Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_standardS_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_mb_SGD_standardS_stderr${TIMESTAMP}.log
```

#### 3. Regularized Models (Lasso, Ridge, ElasticNet)

```bash
./Scripts_bash/run_global2_linear_lasso1.sh > Log_files/Linear/stdout_stderr/global2_linear_lasso1_AlpSelect_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_lasso1_AlpSelect_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_lasso2.sh > Log_files/Linear/stdout_stderr/global2_linear_lasso2_PostAlpSelect_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_lasso2_PostAlpSelect_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_ridge1.sh > Log_files/Linear/stdout_stderr/global2_linear_ridge1_AlpSelect_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_ridge1_AlpSelect_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_ridge2.sh > Log_files/Linear/stdout_stderr/global2_linear_ridge2_PostAlpSelect_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_ridge2_PostAlpSelect_stderr${TIMESTAMP}.log
./Scripts_bash/run_global2_linear_elasticnet2.sh > Log_files/Linear/stdout_stderr/global2_linear_elasticnet2_PostAlpSelect_stdout${TIMESTAMP}.log 2> Log_files/Linear/stdout_stderr/global2_linear_elasticnet2_PostAlpSelect_stderr${TIMESTAMP}.log
```

### B. Decision Trees (Random Forest)

```bash
./Scripts_bash/run_global2_part4and5RF1.sh > Log_files/Decision_Trees/stdout_stderr/global2_part4and5RF_B_unweighted_no_Initial_NotExpanded_292iter_stdout_${TIMESTAMP}.log 2> Log_files/Decision_Trees/stdout_stderr/global2_part4and5RF_B_unweighted_no_Initial_NotExpanded_292iter_stderr_${TIMESTAMP}.log
./Scripts_bash/run_global2_part4and5RF2.sh > Log_files/Decision_Trees/stdout_stderr/global2_part4and5RF_B_unweighted_Expanded_292iter_stdout_${TIMESTAMP}.log 2> Log_files/Decision_Trees/stdout_stderr/global2_part4and5RF_B_unweighted_Expanded_292iter_stderr_${TIMESTAMP}.log
```

### C. XGBoost with AutoML

```bash
# 1 hour
./Scripts_bash/run_global2_part4and5XGBoost_autoFlaml_defaults.sh > Log_files/Decision_Trees/stdout_stderr/global2_part4and5XGBoost_autoFlaml_B_unweighted_1hr_stdout_${TIMESTAMP}.log 2> Log_files/Decision_Trees/stdout_stderr/global2_part4and5XGBoost_autoFlaml_B_unweighted_1hr_stderr_${TIMESTAMP}.log

# 3 hours
./Scripts_bash/run_global2_part4and5XGBoost_autoFlaml_defaults.sh > Log_files/Decision_Trees/stdout_stderr/global2_part4and5XGBoost_autoFlaml_B_unweighted_3hr_stdout_${TIMESTAMP}.log 2> Log_files/Decision_Trees/stdout_stderr/global2_part4and5XGBoost_autoFlaml_B_unweighted_3hr_stderr_${TIMESTAMP}.log
```

### D. Support Vector Regression (SVR)

```bash
./Scripts_bash/run_global2_part4and5SVR1.sh > Log_files/SVR/stdout_stderr/global2_rfb_SVR1_unweighted100_stdout.log 2> Log_files/SVR/stdout_stderr/global2_rfb_SVR1_unweighted100_stderr.log
./Scripts_bash/run_global2_part4and5SVR2.sh > Log_files/SVR/stdout_stderr/global2_rfb_SVR2_unweighted_stdout.log 2> Log_files/SVR/stdout_stderr/global2_rfb_SVR2_unweighted_stderr.log
```

### E. Neural Networks (NN)

#### Very Big Search

```bash
./Scripts_bash/run_global2_NeuralNetwork_very_big_search_weighted_loss1.sh > Log_files/NN/stdout_stderr/global2_very_big_search_weighted_loss1_stdout.log 2> Log_files/NN/stdout_stderr/global2_very_big_search_weighted_loss1_stderr.log
./Scripts_bash/run_global2_NeuralNetwork_very_big_search_weighted_loss2.sh
```

#### Big Search

```bash
./Scripts_bash/run_global2_NeuralNetwork_big_search_weighted_loss1.sh > Log_files/NN/stdout_stderr/global2_big_search_weighted_loss1_stdout.log 2> Log_files/NN/stdout_stderr/global2_big_search_weighted_loss1_stderr.log
./Scripts_bash/run_global2_NeuralNetwork_big_search_weighted_loss2.sh
```

#### Small Search (Weighted and Unweighted)

```bash
./Scripts_bash/run_global2_NeuralNetwork_small_search_weighted_loss1.sh > Log_files/NN/stdout_stderr/global2_small_search_weighted_loss1_stdout.log 2> Log_files/NN/stdout_stderr/global2_small_search_weighted_loss1_stderr.log
./Scripts_bash/run_global2_NeuralNetwork_small_search_weighted_loss2.sh

./Scripts_bash/run_global2_NeuralNetwork_small_search_unweighted_loss1.sh > Log_files/NN/stdout_stderr/global2_small_search_unweighted_loss1_stdout.log 2> Log_files/NN/stdout_stderr/global2_small_search_unweighted_loss1_stderr.log
./Scripts_bash/run_global2_NeuralNetwork_small_search_unweighted_loss2.sh
```

#### Very Small Search

```bash
./Scripts_bash/run_global2_NeuralNetwork_epoch_opt_no_early_stop_weighted_loss1.sh > Log_files/NN/stdout_stderr/global2_epoch_opt_no_early_stop_weighted_loss1_stdout.log 2> Log_files/NN/stdout_stderr/global2_epoch_opt_no_early_stop_weighted_loss1_stderr.log
./Scripts_bash/run_global2_NeuralNetwork_epoch_opt_no_early_stop_weighted_loss2.sh
```


```
