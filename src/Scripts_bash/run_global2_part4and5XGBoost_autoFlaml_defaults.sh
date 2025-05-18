#!/bin/bash
#Train: part5
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."
# echo "Running Expanded Grid Search..."

# # # # ################################################3###This part is for PART2_4 initial run ######################################################
## Common Variables
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count, Weight"
input5='5,101'
input6="None"
input7="unweighted"
input8="BalB"
input9='xgboost_autoFlaml'
## 1st Run Variables

# Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
input3="./Data/train/fold_0_final_train_B_df.pkl,\
./Data/train/fold_1_final_train_B_df.pkl,\
./Data/train/fold_2_final_train_B_df.pkl,\
./Data/train/fold_3_final_train_B_df.pkl,\
./Data/train/fold_4_final_train_B_df.pkl"
# Validation Data: Corresponding validation folds (adjusted to match train folds)
input4="./Data/val/fold_0_final_val_df.pkl,\
./Data/val/fold_1_final_val_df.pkl,\
./Data/val/fold_2_final_val_df.pkl,\
./Data/val/fold_3_final_val_df.pkl,\
./Data/val/fold_4_final_val_df.pkl"
## 2nd Run variables
# Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
input11="./Data/train_val/fold_0_final_train_val_B_df.pkl,\
./Data/train_val/fold_1_final_train_val_B_df.pkl,\
./Data/train_val/fold_2_final_train_val_B_df.pkl,\
./Data/train_val/fold_3_final_train_val_B_df.pkl,\
./Data/train_val/fold_4_final_train_val_B_df.pkl"
# Validation Data: Corresponding validation folds (adjusted to match train folds)
input12="./Data/test/fold_0_final_test_df.pkl,\
./Data/test/fold_1_final_test_df.pkl,\
./Data/test/fold_2_final_test_df.pkl,\
./Data/test/fold_3_final_test_df.pkl,\
./Data/test/fold_4_final_test_df.pkl"


echo "Double Checking the results"

input13='{}'
    # input13='{
    # "n_bins": [128],  
    # "min_samples_leaf": [50],  
    # "min_samples_split": [50],  
    # "n_estimators": [100],  
    # "max_depth": [16],  
    # "max_samples": [0.55, 0.6],  
    # "max_features": ["sqrt"],  
    # "min_impurity_decrease": [0.0]  
    # }'
input10="./Log_files/Decision_Trees, global2_part4_XGBoost_autoFlaml_Defaults_unweighted,./Regression/Results/IntermediateData/Decision_Trees/XGBoost_autoFlaml_Defaults, Random-Search,./Regression/Results/Final_Results/Decision_Trees/XGBoost_autoFlaml_Defaults/, NoRandom-Search,Part4_of_5,Part5_of_5,True,30, logging_file_for_autoFlaml" 
time python ./Scripts_python/global2_part4and5XGBoost_autoML_defaults.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" 

# input10="./Log_files/Decision_Trees, global2_part4_XGBoost_autoML_Defaults_unweighted_part2,./Regression/Results/IntermediateData/Decision_Trees/XGBoost_autoML_Defaults/part2, NoRandom-Search,./Regression/Results/Final_Results/Decision_Trees/XGBoost_autoML_Defaults/, NoRandom-Search,Part4_of_5,Part5_of_5,False"
# time python ./Scripts_python/global2_part4and5XGBoost.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" 