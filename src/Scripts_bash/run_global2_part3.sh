#!/bin/bash
#Train: Part4
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."
########################This part is for PART2_3: Look at REASONABLE SCALERS: #########
input1="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth"
input2="AnnotationScore, ProteinExistence, Predicted_Ontology_F, Predicted_Ontology_P, Depth, Weight"
input3="ols, mb_SGD, lasso, ridge, elasticnet"
# input3="ols"
# input4="./Data/train/fold_0_final_train_A_df.pkl,./Data/train/fold_0_final_train_B_df.pkl"
# input4="./Data/train/fold_0_final_train_A_df.pkl,./Data/train/fold_1_final_train_A_df.pkl, \
#     ./Data/train/fold_2_final_train_A_df.pkl,./Data/train/fold_3_final_train_A_df.pkl, \
#     ./Data/train/fold_4_final_train_A_df.pkl,./Data/train/fold_0_final_train_B_df.pkl, \
#     ./Data/train/fold_1_final_train_B_df.pkl,./Data/train/fold_2_final_train_B_df.pkl, \
#     ./Data/train/fold_3_final_train_B_df.pkl,./Data/train/fold_4_final_train_B_df.pkl"

input4="./Data/train/fold_0_final_train_B_df.pkl, \
    ./Data/train/fold_1_final_train_B_df.pkl,./Data/train/fold_2_final_train_B_df.pkl, \
    ./Data/train/fold_3_final_train_B_df.pkl,./Data/train/fold_4_final_train_B_df.pkl"

input5="weighted,unweighted" #not a identification, but just a direction to choose
input6="BalA, BalB"
input7="./Log_files, global2_part3,./Results/IntermediateData/Linear_Regression/,Part2_3/, "
# input9="RobustScaler, StandardScaler, PowerTransformer, MaxAbsScaler, MinMaxScaler, QuantileTransformer, LogScaler"
# input8="MaxAbsScaler, QuantileTransformer, LogScaler"
input8="MaxAbsScaler, QuantileTransformer, LogScaler,MinMaxScaler, RobustScaler, PowerTransformer, StandardScaler"
time python ./Scripts_python/global2_part3.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6" "$input7" "$input8"
wait 
sleep 30
echo "Global_Part2_Part3 is done"



