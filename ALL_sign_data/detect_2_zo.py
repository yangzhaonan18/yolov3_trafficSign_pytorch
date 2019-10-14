import os 

import torch
import torchvision
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import Lenet5, my_resnt18, FashionCNN

from tqdm import tqdm

from resnet import ResNet18


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


classes = len(os.listdir("/headless/Desktop/yzn_file/code/PyTorch-YOLOv3-Tinghua100K/ALL_sign_data/ALL_data_in_2_train/all_crop_data/"))

# weights_path = "checkpoints_4/model_acc_92__calss_115_epoch_29.pt"
weights_path = "checkpoints_4/model_acc_100__calss_7_epoch_43.pt"
path = "/headless/Desktop/img_crop_2_classification_Tinghua_weights_11/"
save_path  = "/headless/Desktop/img_crop_2_classification_Tinghua_weights_11_2_zo/"



os.environ['CUDA_VISIBLE_DEVICES']='3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs("output", exist_ok=True)

# model = FashionCNN(classes)

model = ResNet18(classes)

model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()


dirs = os.listdir(path)
for dir_ in dirs:
    # detect
    names = os.listdir(os.path.join(path, dir_)) 
    for name in tqdm(names):
        try:
            img_path = path + "/" + dir_ + "/" + name
            input_img_o = Image.open(img_path)
            # print("input_img.shape = ")

            test_transform = torchvision.transforms.Compose([ 
                torchvision.transforms.Resize((28, 28), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                ])

            input_img = test_transform(input_img_o).unsqueeze(0)
            # input_img = torch.autograd.Variable(input_img)

            # print("input_img = ", input_img.size())
            with torch.no_grad():
                pred_class = model(input_img.to(device))
            sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]


            classes = load_classes("ALL_data_in_2_train/names.txt")
            sign_type = classes[int(sign_type)]

            if not os.path.exists(os.path.join(save_path, str(sign_type))):
                os.makedirs(os.path.join(save_path, str(sign_type)))
            input_img_o.save(save_path + "/" + str(sign_type) + "/" + name)

            print("pred_class = ", sign_type)
        except:
            pass

