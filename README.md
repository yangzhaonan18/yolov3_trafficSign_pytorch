# PyTorch-YOLOv3


## Installation
### Clone 

$ git clone https://github.com/yangzhaonan18/yolov3_trafficSign_pytorch
   
    
### Download pretrained weights
1. Download detection(yolov3) weights:
Baidu network disk link: 
download  and put it at: checkpoints/yolov3_ckpt_33.pth

2. Download classifier(CNN) weights:
Baidu network disk link: 
download  and put it at: ALL_sign_data/checkpoints_4/model_acc_94__calss_115_epoch_14.pt

### Detect and classify an image

$ python detection_and_classification.py

detect the images in "./image_for_detect/Tinghua100K/", and the images with results will be saved in "./output",
the "Tinghua100K_result.json" result will be saved in "./result/"


### How to test the results?



## How to train my dataset?
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


