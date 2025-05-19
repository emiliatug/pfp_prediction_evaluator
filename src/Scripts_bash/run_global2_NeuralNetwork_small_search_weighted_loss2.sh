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
input9='neural_network_manual_tuning'

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
input10="./Log_files/NN, global2_small_search_weighted_loss2,./Regression/Results/Final_Results/NN/small_search_weighted, weighted_loss, small_search,./ResultsIntermediateData/NN/manual_tuning/small_search_weighted/models"
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
    "1": [256],
    "2": [256, 128],
    "3": [256, 128, 64],
    "4": [256, 128, 64, 32],
    "5": [256, 128, 64, 32, 16],
    "6": [256, 128, 64, 32, 16, 8],
    "7": [256, 160, 128, 64, 32, 16, 8],
    "8": [256, 192, 128, 96, 64, 32, 16, 8],
    "9": [256, 192, 128, 104, 64, 48, 32, 16, 8]                                          
}'
input14="Adam,Adagrad,RMSprop"
# input14="Adam,Adagrad"
# input15="1"
input15="1,2,3,4,5,6,7,8,9"
# input15="1"


time python ./Scripts_python/global2_NeuralNetwork_any_search_weighted_loss2.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8" "$input9" "$input10" "$input11" "$input12" "$input13" "$input14" "$input15" 
wait 
sleep 30
