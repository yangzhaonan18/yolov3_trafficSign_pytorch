import os 
import numpy as np
# from sklearn.model_selection import train_test_split



# from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torchvision
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

#  my py files

from model import Network, Lenet5, my_resnt18, FashionCNN


from resnet import ResNet18


from utils.dataset import MyDataset
from utils.show import print_accuracy
 

np.random.seed(0)
# classes =  len(os.listdir("/headless/Desktop/yzn_file/code/PyTorch-YOLOv3-Tinghua100K/ALL_sign_data/ALL_data_in_2_train/all_crop_data/"))
classes = 115
epochs  = 50
batch_size = 64  # 64
learning_rate = 0.002  # 0.002
log_interval = 100
input_size = 28 * 28 * 3  # for "Network" model
train_ratio = 0.8 


import os
os.environ['CUDA_VISIBLE_DEVICES']='4'


writer = SummaryWriter("log_LetNet_5")
# writer = SummaryWriter("log_fashionModel_2")

info_file = "ALL_data_in_2_train/info.txt"
names = "ALL_data_in_2_train/names.txt"

pre_train = False
weights_path = "checkpoints/model_acc_92__calss_112_epoch_14.pt"





# train_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#     # transforms.Grayscale(num_output_channels=1),  #  to gray
#     # transforms.Resize((30, 30), interpolation=2),
#     # # transforms.CenterCrop((28, 28)),
#     # transforms.RandomCrop((28, 28), padding=5, pad_if_needed=True),
  
#     torchvision.transforms.RandomRotation(15),
#     torchvision.transforms.Resize((35, 35), interpolation=2),
#     torchvision.transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(0.9, 1.1), interpolation=2),
    
#      # torchvision.transforms.Resize((28, 28), interpolation=2),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # mean=(0.5,), std=(0.5,))
#     ])

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # transforms.Grayscale(num_output_channels=1),  #  to gray
    # transforms.Resize((30, 30), interpolation=2),
    # transforms.CenterCrop((28, 28)),
    # transforms.RandomCrop((28, 28), padding=5, pad_if_needed=True),
  
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.Resize((35, 35), interpolation=2),
    torchvision.transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(0.9, 1.1), interpolation=2),
    
    # torchvision.transforms.Resize((28, 28), interpolation=2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # mean=(0.5,), std=(0.5,))
    ])




dataset = MyDataset(info_file, names, train_transform)


data_num = len(dataset)
indices = list(range(data_num))
np.random.shuffle(indices)
split = int(np.floor(train_ratio * data_num))

train_idx, test_idx = indices[:split], indices[split:]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)


train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    # shuffle=True, # "sampler" option is mutually exclusive with "shuffle"
    num_workers=16,
    pin_memory=True,
    sampler=train_sampler,
)

test_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    num_workers=16,
    pin_memory=True,
    sampler=test_sampler
    )


# ####### model for experiments ############## 
# model=Network()
# 
# model = my_resnt18(classes)
# model = Lenet5(classes)
# model = FashionCNN(classes)

model = ResNet18(classes)



# optimizer=torch.optim.SGD(model.parameters(), lr=0.03, weight_decay= 1e-6, momentum = 0.87,nesterov = True)
# optimizer=torch.optim.SGD(model.parameters(),lr=0.003,)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if pre_train:
    model.load_state_dict(torch.load(weights_path))    

model.to(device)
criterion.to(device)

# if torch.cuda.is_available():
#     model.cuda()
#     criterion.cuda()



#  train and test 
flag = 0
for epoch in range(epochs):
    class_correct = list(0. for i in range(classes))
    class_total = list(0. for i in range(classes))

    train_loss = 0
    model.train()
    preds = []
    targets = []
    for batch_i, (path, img, target) in enumerate(train_loader):
        image = img.clone()
         
        # image = transforms.ToPILImage()(torch.from_numpy(image.numpy()))
        
        # plt.figure(figsize=(4, 4))
        # plt.ion()
        # plt.axis('off')   

        # image_1 = image[0].squeeze(0).permute(1, 2, 0)  
    
        # image_1[:, :, 0], image_1[:, :, 1], image_1[:, :, 2] = 
        #   image_1[:, :, 1], image_1[:, :, 2], image_1[:, :, 0]
        # plt.imshow(image_1, cmap="gray")

        # ##### show image  ##############
        # number = 64 # len(path)
        # fig, ax = plt.subplots(9, 16, sharex=True, sharey=True)
        # for i in range(number):
        #     row = i // 16
        #     col = i % 16
        #     org_img = Image.open(path[i])
        #     org_img = org_img.resize((28, 28))
        #     transf_img = image[i].squeeze(0).permute(1, 2, 0)

        #     ax[row][col].imshow(org_img)
        #     ax[row + 5][col].imshow(transf_img)
              # ax[i,j].set_axis_off()
        #     ax[0][0].set_title("original images:")
        #     ax[5][0].set_title("transforms images:")
        # plt.axis("off") 
        # plt.show(0.01)

       
       
        img = torch.autograd.Variable(img).to(device)
        target = torch.autograd.Variable(target).to(device)

        optimizer.zero_grad()
        prediction = model(img)
        target = target.squeeze()
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # train loss
        # if batch_i % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_i * len(img), len(dataloader.dataset),
        #         100. * batch_i / len(dataloader), loss.data))
          



        _, pred = torch.max(prediction, 1)  #  class
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))  # from class to  0 1 
        # calculate test accuracy for each object class
        for i in range(len(correct)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1 

        preds.extend(pred.cpu())
        targets.extend(target.cpu())
    print("train_accuracy:")
    print_accuracy(targets, preds)  #  show results


    train_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f"\nEpoch: {epoch}", 
          f"Train Accuracy: {int(train_accuracy)}%" ,
          f"{int(np.sum(class_correct))}/{int(np.sum(class_total))}",
        )

    

    # #########  for test ################
    model.eval()
    
    with torch.no_grad():
        preds = []
        targets = []
        test_loss = 0
        correct = 0
        for i, (_, image, target) in enumerate(test_loader):

            image = image.to(device) 
            target = target.to(device)
            prediction = model(image)
            
            target = target.squeeze(0)
            test_loss += criterion(prediction, target)
            _, pred = torch.max(prediction, 1)
            correct += pred.eq(target).sum().item()

            preds.append(pred.cpu())
            targets.append(target.cpu())
        print("test_accuracy:")
        print_accuracy(targets, preds)  #  show results



    cur_acc = correct / len(test_loader)


    test_accuracy = int(100 * correct / len(test_loader))

    # if test_accuracy > 91  and flag == 0:
        # flag = 1
    torch.save(model.state_dict(), f'checkpoints/model_acc_{test_accuracy}__calss_{classes}_epoch_{epoch}.pt')
        # break #  stop training


    print(f"Epoch: {epoch},",
        f"Test Accuracy: {test_accuracy}%",
        f"{int(np.sum(correct))}/{len(test_loader)}",
        f"\t\t Train loss:{train_loss / len(train_loader):.1f}",
        f"Test loss:{test_loss / len(test_loader):.1f}",
        )


    writer.add_scalars('data/scalar', { "train_accuracy": train_accuracy,
                                        "test_accuracy": test_accuracy,
                                        }, epoch)

    # model.eval()    



        # running_loss+=loss.item()
# training


# test 




# model.load_state_dict(torch.load('model.pt'))




