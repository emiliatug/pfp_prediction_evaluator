#!/bin/bash
#Train: part5
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."

# # # # ################################################3###This part is for PART2_4 and PART5 LASSO ######################################################
## Common Variables
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count, Weight"
input5='5,101'
input6="LogScaler,None"
input7="unweighted"
input8="BalB"
input9='lasso'

## 1st Run Variable
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
input10="./Log_files/Linear, global2_linear_lasso2_post_alpha_selection,./Regression/Results/IntermediateData/Linear_Regression/lasso/Part2_4, Grid-Search,./Regression/Results/Final_Results/Linear_Regression/lasso/, NoGrid-Search,Part4_of_5,Part5_of_5,NoAlpha-Selection" 
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
input13='{
 "alpha":[
 1.62975083e-04, 3.25950167e-05, 6.30957344e-05, 8.23738707e-05,
 1.07542208e-04, 1.40400425e-04, 1.83298071e-04, 2.39302573e-04,
 3.12418571e-04, 4.07874276e-04, 5.32495313e-04, 6.95192796e-04,
 9.07600522e-04, 1.18490685e-03, 1.54694077e-03, 2.01958975e-03,
 2.63665090e-03, 3.44224760e-03, 4.49398459e-03, 5.86706707e-03,
 7.65967823e-03, 1.00000000e-02]}'


time python ./Scripts_python/global2_linear_regr_sgd_lass_ridge_elastic.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" 
wait 
sleep 30