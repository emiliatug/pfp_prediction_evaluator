#!/bin/bash
#Train: part5
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."

# # # # ################################################3###This part is for PART2_4 and PART5 ridge ######################################################
## Common Variables
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth,Total_Count, Weight"
input5='5,101'
input6="LogScaler,None"
input7="unweighted"
input8="BalB"
input9='ridge'

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
input10="./Log_files/Linear, global2_linear_ridge2_post_alpha_selection,./Regression/Results/IntermediateData/Linear_Regression/ridge/Part2_4, Grid-Search,./Regression/Results/Final_Results/Linear_Regression/ridge/, NoGrid-Search,Part4_of_5,Part5_of_5,NoAlpha-Selection" 
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
 0.01519911, 0.00303982, 0.00063096, 0.00082374, 0.00107542, 0.001404,
 0.00183298, 0.00239303, 0.00312419, 0.00407874, 0.00532495, 0.00695193,
 0.00907601, 0.01184907, 0.01546941, 0.0201959, 0.02636651, 0.03442248,
 0.04493985, 0.05867067, 0.07659678, 0.1]}'

time python ./Scripts_python/global2_linear_regr_sgd_lass_ridge_elastic.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" 
wait 
sleep 30