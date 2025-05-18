#!/bin/bash
#Train: Part4
# Move to the project root (one level up from Scripts_bash)
cd "$(dirname "$0")/.."
########################This part is for PART2_1: Generate A and B versions of train data and move the val/test files from Global1 to Global2: #########
input1="../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train/part6/fold_0_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train/part6/fold_1_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train/part6/fold_2_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train/part6/fold_3_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train/part6/fold_4_train/final_train_df.pkl,\
    ./Data/train/"
input2="../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train_val/part6/fold_0_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train_val/part6/fold_1_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train_val/part6/fold_2_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train_val/part6/fold_3_train/final_train_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/train_val/part6/fold_4_train/final_train_df.pkl, \
    ./Data/train_val/"
input3="../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/val/part456val/fold_0_val/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/val/part456val/fold_1_val/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/val/part456val/fold_2_val/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/val/part456val/fold_3_val/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/val/part456val/fold_4_val/final_val_df.pkl,\
    ./Data/val/"
input4="../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/test/part456test/fold_0_test/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/test/part456test/fold_1_test/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/test/part456test/fold_2_test/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/test/part456test/fold_3_test/final_val_df.pkl, \
../Global_Part1_Data_Generation/Data/Post_CV_Split_Data/test/part456test/fold_4_test/final_val_df.pkl, \
    ./Data/test/"
input5="fold_0,fold_1,fold_2,fold_3,fold_4"
input6="./Log_files, global2_part1"
time python ./Scripts_python/global2_part1.py "$input1" "$input2" "$input3" "$input4" "$input5" "$input6"
wait 
sleep 30
echo "Global_Part2_Part1 is done"

########################This part is for PART2_2: Generating Naive Models:###################################################################################
input1="./Data/train_val/fold_0_final_train_val_A_df.pkl,./Data/train_val/fold_0_final_train_val_B_df.pkl,\
./Data/train_val/fold_1_final_train_val_A_df.pkl,./Data/train_val/fold_1_final_train_val_B_df.pkl,\
./Data/train_val/fold_2_final_train_val_A_df.pkl,./Data/train_val/fold_2_final_train_val_B_df.pkl,\
./Data/train_val/fold_3_final_train_val_A_df.pkl,./Data/train_val/fold_3_final_train_val_B_df.pkl,\
./Data/train_val/fold_4_final_train_val_A_df.pkl,./Data/train_val/fold_4_final_train_val_B_df.pkl"
input2="./Data/test/fold_0_final_test_df.pkl,./Data/test/fold_0_final_test_df.pkl,./Data/test/fold_1_final_test_df.pkl,\
./Data/test/fold_1_final_test_df.pkl,./Data/test/fold_2_final_test_df.pkl,./Data/test/fold_2_final_test_df.pkl,\
./Data/test/fold_3_final_test_df.pkl,./Data/test/fold_3_final_test_df.pkl,./Data/test/fold_4_final_test_df.pkl,\
./Data/test/fold_4_final_test_df.pkl"
input3="NaiveMeanModel_A,NaiveMeanModel_B"
input4="./Results/Final_Results/naive_models,./Log_files,global2_part2_naive_models"
time python ./Scripts_python/global2_part2.py "$input1" "$input2" "$input3" "$input4"
wait 
sleep 30
echo "Global_Part2_Part2 is done"
