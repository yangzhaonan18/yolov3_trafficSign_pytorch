# PyTorch-YOLOv3


## 1. Installation
### git clone
```python
git clone https://github.com/yangzhaonan18/yolov3_trafficSign_pytorch
```  
    
### Download pretrained weights
1. Download detection(yolov3) weights:
Baidu network disk link: 
download  and put it at: ./checkpoints/yolov3_ckpt_33.pth

2. Download classifier(CNN) weights:
Baidu network disk link:
download  and put it at: ./ALL_sign_data/checkpoints_4/model_acc_94__calss_115_epoch_14.pt

### Detect and classify an image
```python
python3 detection_and_classification.py
```
detect the images in "./image_for_detect/Tinghua100K/", 
and the images with results will be saved in "./output", the "Tinghua100K_result.json" result will be saved in "./result/"


## 2. How to test the results?
```
cd Tinghua100K_data/python/
python3 my_result_classes.py

```

results is:
```
iou:0.5, size:[0,400), types:[w55, ...total 53...], accuracy:0.0, recall:0.0
iou:0.5, size:[0,32), types:[w55, ...total 53...], accuracy:0.0, recall:0.0
iou:0.5, size:[32,96), types:[w55, ...total 53...], accuracy:0.0, recall:0.0
iou:0.5, size:[96,400), types:[w55, ...total 53...], accuracy:1, recall:1
iou:0.5, size:[0,400), types:w55, accuracy:1, recall:1
......
......
```

## 3. How to train my dataset?
### Train YOLOv3 detection
1. Download pretrained weights(on COCO): darknet53.conv.74 
$ cd ./weights
$ bash download_weights.sh

2. 

### Train CNN classifier 
1. Download traffic sign data to train classifier
Baidu network disk link: https://pan.baidu.com/s/133wOElvWHn0Fm4RzOGLk3w
and unzip it in LL_sign_data/ALL_data_in_2_train/


## how to test my dataset?

## how to train your dataset?


