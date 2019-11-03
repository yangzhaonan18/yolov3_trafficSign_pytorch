# PyTorch-YOLOv3


## 1. Installation
### 1.1. git clone
```python
git clone https://github.com/yangzhaonan18/yolov3_trafficSign_pytorch
```  
    
### 1.2. Download pretrained weights
1. Download detection(yolov3) weights:
Baidu network disk link(234MB): https://pan.baidu.com/s/1BWySwi22nsFTB7-c0ualZA

download  and put it at: ./checkpoints/yolov3_ckpt_33.pth

2. Download classifier(CNN) weights:
Baidu network disk link(43MB): https://pan.baidu.com/s/1Id65qVFrAp-S5G--57LG2Q

download  and put it at: ./ALL_sign_data/checkpoints/

### 1.3. Detect and classify an image
```python
python3 detection_and_classification.py
```
detect the images in "./image_for_detect/Tinghua100K/", 
and the images with results will be saved in "./output", the "Tinghua100K_result.json" result will be saved in "./result/"

images with results:
<p align="center">
  <img width="1000" src="image_for_github/00085.png">
</p>

## 2. How to test the results?
```
cd Tinghua100K_data/python/
python3 my_result_classes.py

```

results is:
```
iou:0.5, size:[0,400), types:[w55, ...total 53...], accuracy:0.8845589434208381, recall:0.9304519337964154
iou:0.5, size:[0,32), types:[w55, ...total 53...], accuracy:0.8160990712074303, recall:0.885752688172043
iou:0.5, size:[32,96), types:[w55, ...total 53...], accuracy:0.9297398348652438, recall:0.9740492900277461
iou:0.5, size:[96,400), types:[w55, ...total 53...], accuracy:0.9261477045908184, recall:0.8672897196261682
iou:0.5, size:[0,400), types:w55, accuracy:0.9611650485436893, recall:0.908256880733945
iou:0.5, size:[0,400), types:p27, accuracy:0.9156626506024096, recall:0.9047619047619048
iou:0.5, size:[0,400), types:il80, accuracy:0.9746192893401016, recall:0.9746192893401016
iou:0.5, size:[0,400), types:i1, accuracy:0.8571428571428571, recall:1.0
iou:0.5, size:[0,400), types:il100, accuracy:0.89, recall:0.967391304347826
......
......
```

## 3. How to train my dataset?
### 3.1. Train YOLOv3 detection
1. Download pretrained weights(on COCO): darknet53.conv.74 
```
cd ./weights
bash download_weights.sh
```


2. Train YOLOv3 detection

```
cd ../
python3 train.py --data_config config/Tinghua100K.data --pretrained_weights weights/darknet53.conv.74
```

The YOLOv3 training weights will be saved in ./checkpoints/
### 3.2. Train CNN classifier 
1. Download traffic sign data to train classifier
Baidu network disk link: https://pan.baidu.com/s/133wOElvWHn0Fm4RzOGLk3w
and unzip it in ALL_sign_data/ALL_data_in_2_train/

```
cd ./ALL_sign_data/
bash  run.sh
```

The train weights will be saved in ./ALL_sign_data/checkpoints


## 4. how to train your dataset?



