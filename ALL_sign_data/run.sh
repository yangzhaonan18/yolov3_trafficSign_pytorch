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
echo "    "
echo "    "