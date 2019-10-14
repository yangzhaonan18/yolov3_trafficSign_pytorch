#!/bin/bash

cd ./ALL_data_in_2_train
echo "    "
echo "    "
echo "    "
echo "    "
echo "create info.txt and name.txt for sign data ........"
python createPath.py
cd ..

echo "start  training ......."
python classification.py
echo "finish training, when test accuracy greater than 91%, save a weight and stop training"
echo "weight save in : " 
echo "/headless/Desktop/yzn_file/code/PyTorch-YOLOv3-Tinghua100K/ALL_sign_data/model_acc_90__epoch_3.pt"
echo "    "
echo "    "