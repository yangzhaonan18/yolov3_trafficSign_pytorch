import os 

import torch
import torchvision
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from resnet import ResNet18

from model import Lenet5, my_resnt18, FashionCNN


# classes = 145
weights_path = "model_acc_99__calss_7_epoch_26.pt"
# img_path = "all_data_noUse/GTSDB_JPG/18/00005.jpg"
# img_path = "data_emotion/CK+last5-raw/Angry/S010/S010_004_00000015.png"
# types = [Angry_, Contempt_, Disgust_, Fear_, Happy_, Sad_, Surprise_]


os.environ['CUDA_VISIBLE_DEVICES']='3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs("output", exist_ok=True)

# model = FashionCNN(classes)

model = ResNet18()

model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()

path = "data_emotion/CK+last5-raw"
emotions = os.listdir(path)
emotions = sorted(emotions)





# cAP_video = [0 for ]

correct_img_num = 0
total_img_num = 0
AP_img = 0  #  

correct_video_num = 0
total_video_num = 0 
AP_video = 0  

 

c_correct_img_num = [0 for i in range(7)]
c_total_img_num = [0 for i in range(7)]
cAP_img = [0 for i in range(7)]


c_correct_video_num = [0 for i in range(7)]
c_total_video_num = [0 for i in range(7)]
cAP_video = [0 for i in range(7)]


def max_count(lt):
    d = {}
    max_key = None
    for i in lt:
        if i not in d:
            count = lt.count(i)
            d[i] = count
            if count > d.get(max_key, 0):
                max_key = i
    return max_key


for i, emotion in enumerate(emotions):
  
    gt_emotion_type = i
    peoples = os.listdir(path + "/" + emotion)
    print("emotion = ", emotion)
    for people in peoples:
        total_video_num += 1
        
        
        names = os.listdir(path + "/" + emotion + "/" + people)
        for k, name in enumerate(names):
            total_img_num += 1
            people_types = []
            img_path = path + "/" + emotion + "/" + people  + "/" + name
            

            # detect 
            input_img = Image.open(img_path)
            # print("input_img.shape = ")

            test_transform = torchvision.transforms.Compose([ 
                torchvision.transforms.Grayscale(num_output_channels=1), 
                torchvision.transforms.Resize((28, 28), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                ])

            input_img = test_transform(input_img).unsqueeze(0)
            # input_img = torch.autograd.Variable(input_img)

            # print("input_img = ", input_img.size())
            with torch.no_grad():
                pred_class = model(input_img.to(device))
            pred_emotion_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]


            # print("img_path = ", img_path)
            # print("gt_emotion_type = ", gt_emotion_type)
            # print("pred_emotion_type = ", pred_emotion_type)
            people_types.append(pred_emotion_type)
            
            if gt_emotion_type == pred_emotion_type:
                correct_img_num += 1
                c_correct_img_num[i] += 1
            
            c_total_img_num[i] += 1
            if k == 4:
                c_total_video_num[i] += 1
                people_type = max_count(people_types)
                if gt_emotion_type == people_type:
                    c_correct_video_num[i] += 1
                    correct_video_num += 1





AP_img = correct_img_num / total_img_num
AP_video = correct_video_num / total_video_num

for i in range(7):
    cAP_video[i] = c_correct_video_num[i] / c_total_video_num[i]



for i in range(7):
    cAP_img[i] = c_correct_img_num[i] / c_total_img_num[i]

print("\n") 
print(correct_img_num , total_img_num)
print("AP_img = ", AP_img)

print("\n") 
print(correct_video_num , total_video_num)
print("AP_video = ", AP_video)


print("\n")
print(c_correct_img_num, c_total_img_num)
print("cAP_img = ", cAP_img)

print("\n")
print(c_correct_video_num,  c_total_video_num)
print("cAP_video = ", cAP_video)





















