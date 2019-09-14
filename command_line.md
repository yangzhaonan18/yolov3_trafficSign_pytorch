
## YOlOv3
https://github.com/eriklindernoren/PyTorch-YOLOv3

###  train
$ python3 train.py --data_config config/Tinghua100K.data --pretrained_weights weights/darknet53.conv.74

$ python3 train.py --data_config config/Tinghua100K.data --pretrained_weights weights/darknet53.conv.74

$ python3 train.py --data_config config/Tinghua100K.data --pretrained_weights checkpoints/yolov3_ckpt_184.pth

### test
 python3 test.py --weights_path weights/yolov3.weights
 python3 test.py --weights_path checkpoints/yolov3_ckpt_99.pth


### detect
python3 detect.py --image_folder data/samples/ 
python3 detect.py --image_folder data/samples/  --weights_path checkpoints/yolov3_ckpt_129.pth 
python3 detect.py --image_folder data/samples/  --model_def config/yolov3_2_11_notChange.cfg --weights_path checkpoints_28/yolov3_ckpt_129.pth

### evaluation_to_get_figure
python evaluation_to_get_figure.py 