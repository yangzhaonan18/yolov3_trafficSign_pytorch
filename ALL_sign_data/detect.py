import os 

import torch
import torchvision
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from resnet import ResNet18

from model import Lenet5, my_resnt18, FashionCNN
 

classes = 115

weights_path = "checkpoints_4/model_acc_94__calss_115_epoch_14.pt"
img_path = "all_data_noUse/GTSDB_JPG/18/00005.jpg"



os.environ['CUDA_VISIBLE_DEVICES']='3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs("output", exist_ok=True)

model = FashionCNN(classes)
# model = ResNet18(classes)


model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()

# detect 
input_img = Image.open(img_path)
# print("input_img.shape = ")

test_transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize((28, 28), interpolation=2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

input_img = test_transform(input_img).unsqueeze(0)
# input_img = torch.autograd.Variable(input_img)

# print("input_img = ", input_img.size())
with torch.no_grad():
    pred_class = model(input_img.to(device))
sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()



print("pred_class = ", sign_type)







