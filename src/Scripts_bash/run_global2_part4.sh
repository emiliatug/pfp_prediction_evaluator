#!/bin/bash
#Train: Part4
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."
################################################3###This part is for PART2_4:OLS: BalA and Weighted######################################################
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth, Weight"
# Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
input3="./Data/train_val/fold_0_final_train_val_A_df.pkl,\
./Data/train_val/fold_1_final_train_val_A_df.pkl,\
./Data/train_val/fold_2_final_train_val_A_df.pkl,\
./Data/train_val/fold_3_final_train_val_A_df.pkl,\
./Data/train_val/fold_4_final_train_val_A_df.pkl"
# Validation Data: Corresponding validation folds (adjusted to match train folds)
input4="./Data/val/fold_0_final_test_A_df.pkl,\
./Data/test/fold_1_final_test_A_df.pkl,\
./Data/test/fold_2_final_test_A_df.pkl,\
./Data/test/fold_3_final_test_A_df.pkl,\
./Data/test/fold_4_final_test_A_df.pkl"
input5=5
input6="./Results/IntermediateData/Linear_Regression/ols/Part2_3/,best_df_BalA.pkl,LogScaler"
input7="weighted"
input8="BalA"
input9='ols'
input10="./Log_files, global2_part4_ols_A_weighted,./Final_Results/Linear_Regression/ols/" # ./Results/IntermediateData/Linear_Regression/ols/Part2_4"
time python ./Scripts_python/global2_part4.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" 
wait 
sleep 30  

# # # ################################################3###This part is for PART2_4:OLS: BalA and UNWeighted######################################################
# input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth"
# input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth, Weight"
# # Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
# input3="./Data/train_val/fold_0_final_train_val_A_df.pkl,\
# ./Data/train_val/fold_1_final_train_valA_df.pkl,\
# ./Data/train_val/fold_2_final_train_valA_df.pkl,\
# ./Data/train_val/fold_3_final_train_val_A_df.pkl,\
# ./Data/train_val/fold_4_final_train_val_A_df.pkl"
# # Validation Data: Corresponding validation folds (adjusted to match train folds)
# input4="./Data/test/fold_0_final_test_A_df.pkl,\
# ./Data/test/fold_1_final_test_A_df.pkl,\
# ./Data/test/fold_2_final_test_A_df.pkl,\
# ./Data/test/fold_3_final_test_A_df.pkl,\
# ./Data/test/fold_4_final_test_A_df.pkl"
# input5=5
# input6="./Results/IntermediateData/Linear_Regression/ols/Part2_3/,best_df_BalA.pkl,LogScaler"
# input7="unweighted"
# input8="BalA"
# input9='ols'
# input10="./Log_files, global2_part4_ols_A_unweighted,./Final_Results/Linear_Regression/ols/" # ./Results/IntermediateData/Linear_Regression/ols/Part2_4"
# time python ./Scripts_python/global2_part4.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" 
# wait 
# sleep 30

# # # ################################################3###This part is for PART2_4:OLS: BalB and Weighted######################################################
# input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth"
# input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth, Weight"
# # Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
# input3="./Data/train_val/fold_0_final_train_val_A_df.pkl,\
# ./Data/train_val/fold_1_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_2_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_3_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_4_final_train_val_B_df.pkl"
# # Validation Data: Corresponding validation folds (adjusted to match train folds)
# input4="./Data/test/fold_0_final_test_B_df.pkl,\
# ./Data/test/fold_1_final_test_B_df.pkl,\
# ./Data/test/fold_2_final_test_B_df.pkl,\
# ./Data/test/fold_3_final_test_B_df.pkl,\
# ./Data/test/fold_4_final_test_B_df.pkl"
# input5=5
# input6="./Results/IntermediateData/Linear_Regression/ols/Part2_3/,best_df_BalB.pkl,LogScaler"
# input7="weighted"
# input8="BalB"
# input9='ols'
# input10="./Log_files, global2_part4_ols_B_weighted,./Final_Results/Linear_Regression/ols/" # ./Results/IntermediateData/Linear_Regression/ols/Part2_4"
# time python ./Scripts_python/global2_part4.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" 
# wait 
# sleep 30

# # # ################################################3###This part is for PART2_4:OLS: BalB and UNWeighted######################################################
# input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth"
# input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth, Weight"
# # Train Data: Folds structured as (fold0A, fold0B, fold1A, fold1B, ..., fold5A, fold5B)
# input3="./Data/train_val/fold_0_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_1_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_2_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_3_final_train_val_B_df.pkl,\
# ./Data/train_val/fold_4_final_train_val_B_df.pkl"
# # Validation Data: Corresponding validation folds (adjusted to match train folds)
# input4="./Data/test/fold_0_final_test_B_df.pkl,\
# ./Data/test/fold_1_final_test_B_df.pkl,\
# ./Data/test/fold_2_final_test_B_df.pkl,\
# ./Data/test/fold_3_final_test_B_df.pkl,\
# ./Data/test/fold_4_final_test_B_df.pkl"
# input5=5
# input6="./Results/IntermediateData/Linear_Regression/ols/Part2_3/,best_df_BalB.pkl,LogScaler"
# input7="unweighted"
# input8="BalB"
# input9='ols'
# input10="./Log_files, global2_part4_ols_B_unweighted,./Final_Results/Linear_Regression/ols/" # ./Results/IntermediateData/Linear_Regression/ols/Part2_4"
# time python ./Scripts_python/global2_part4.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" 
# wait 
# sleep 30

# # echo "Global_Part2_Part3 is done"

