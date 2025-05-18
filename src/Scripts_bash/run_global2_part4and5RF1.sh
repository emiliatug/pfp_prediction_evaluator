#!/bin/bash
#Train: part5
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."
echo "Running Initial_NotExpanded Grid Search..."

# # # # ################################################3###This part is for PART2_4 initial run ######################################################
## Common Variables
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count, Weight"
input5='5,101'
input6="None" #None to not do scaling, and LogScaler to have the scaling done
input7="unweighted"
input8="BalB"
input9='random_forest' #after that it will be xg boost

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
input10="./Log_files/Decision_Trees, global2_part4_Random_Forest_Forest_unweighted_Initial_NotExpanded,./Regression/Results/IntermediateData/Decision_Trees/Random_Forest/Part2_4/Initial_NotExpanded, Random-Search,./Regression/Results/Final_Results/Decision_Trees/Random_Forest/, NoRandom-Search,Part4_of_5,Part5_of_5,True" 
input13='{  
    "n_bins": [128, 256],  
    "min_samples_leaf": [50,500, 1000],  
    "min_samples_split": [50,500, 1000],  
    "n_estimators": [50,500, 1000], 
    "max_depth": [8,16], 
    "max_samples": [0.6, 0.75, 1.0],  
    "max_features": ["sqrt", "log2",0.75],  
    "min_impurity_decrease": [0.0, 0.0001, 0.001]  
}' 
# #"max_depth": [6,16] in the old one
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

time python ./Scripts_python/global2_part4and5RF.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13"

echo "Initial_NotExpanded Grid Search is completed. "
echo ""
echo "Results saved in 'Summer2024/FF_editing_transfer/E_Tugolukov/Global_Part2_ML_not_DL/Results/IntermediateData/Decision_Trees/Random_Forest/Part2_4/dict_best_parameters_values_per_fold.pkl'."
echo ""
echo "Please review 'dict_best_parameters_values_per_fold.pkl' in Jupyter Notebook before continuing, and editing input14"

while true; do
    echo -n "Type 'Continue': " && read user_input #this can be simplified by the **read -p "Type 'Continue' to proceed: " user_input**
    if [ "$user_input" == "Continue" ]; then
        break  # Exit the loop when the correct input is given
    else
        echo "Invalid input. Please type 'Continue' exactly."
    fi
done
# #Input dict# get this as an average expansion of the best parameter of hte 5 CV
# print("Double Checking the results")
# input13='{
#     "n_bins": [128, 256, 384],  
#     "min_samples_leaf": [50],  
#     "min_samples_split": [50],  
#     "n_estimators": [900, 950,975, 1000,1025, 1050, 1100],  
#     "max_depth": [16],  
#     "max_samples": [0.6, 0.75, 1.0],  
#     "max_features": ["sqrt"],  
#     "min_impurity_decrease": [0.0, 0.0001]  
# }'

# input10="./Log_files, global2_part4_Random_Forest_Forest_unweighted_Expanded,./Results/IntermediateData/Decision_Trees/Random_Forest/Part2_4/Expanded, NoRandom-Search,./Results/Final_Results/Decision_Trees/Random_Forest/, NoRandom-Search,Part4_of_5,Part5_of_5,False"

# time python ./Scripts_python/global2_part4and5RF.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" 