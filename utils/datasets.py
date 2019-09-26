import glob
import random
import os
import sys
import numpy as np
from PIL import Image


import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import random


import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib.patches as patches


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


 

def show_img_target(img, targets):
    c, h, w = img.shape
    transf_img = img.squeeze(0).permute(1, 2, 0)

    print(transf_img.shape)
    plt.imshow(transf_img)
    currentAxis=plt.gca()
    for target in targets: 
        x1 = int((target[2] - target[4] / 2) * w)
        y1 = int((target[3] - target[5] / 2) * w)
        w1 = int(target[4] * w)
        h1 = int(target[5] * w)
        rect = patches.Rectangle((x1, y1), w1, h1 , linewidth=1, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)
    plt.show()


def crop_pad_to_size_copy_to_save(img, targets, resize_rat, pad_value):
    c, h, w = img.shape
    
    
    img = F.interpolate(img.unsqueeze(0), size=[int(resize_rat * w), int(resize_rat * w)],
                                    mode="nearest").squeeze(0)
    if resize_rat >= 1:
        img = img[:, :w, :w]
    else:
        diff =  w - int(resize_rat * w)
        # Determine padding
        pad = (0, diff, 0, diff)  
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

    targets[:, 2:] = targets[:, 2:] * resize_rat 

    return img, targets


def crop_pad_to_size(img, targets, resize_rat, pad_value):

    c, h, w = img.shape
   
    img = F.interpolate(img.unsqueeze(0), size=[int(resize_rat * w), int(resize_rat * w)],
                                    mode="nearest").squeeze(0)
 
    if resize_rat >= 1:
        img = img[:, :w, :w]
    else:
        diff =  w - int(resize_rat * w)
        # Determine padding
        pad = (0, diff, 0, diff)  
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

    targets[:, 2:] = targets[:, 2:] * resize_rat 

    # c, h, w = img.shape
    # min_x1 = torch.min(torch.min(targets[:, 2] - targets[:, 4]/2 )) * w 
    # min_y1 = torch.min(torch.min(targets[:, 3] - targets[:, 5]/2 )) * w 
    # rand_wh = int(random.uniform(0, torch.min(min_x1, min_y1)) )
    # img = img[ : , rand_wh: , rand_wh: ]
    # targets[:, 2:] = targets[:, 2:] * (w / (w - rand_wh))

    return img, targets

    
def random_resize_ratio_2(img, targets, img_size, small_sign, large_sign):  # yzn
    c, h, w = img.shape
    # all position of signs in x1y1 x2y2 
    min_x1 = torch.min(torch.min(targets[:, 2] - targets[:, 4]/2 )) * w 
    min_y1 = torch.min(torch.min(targets[:, 3] - targets[:, 5]/2 )) * w 

    max_x2 = torch.max(torch.max(targets[:, 2] + targets[:, 4]/2 )) * w 
    max_y2 = torch.max(torch.max(targets[:, 3] + targets[:, 5]/2 )) * w   

    crop_min_h = max_y2 - min_y1
    crop_min_w = max_x2 - min_x1
    # print("crop_min_h, crop_min_w = ", crop_min_h, crop_min_w)
    crop_min_size = torch.max(crop_min_h, crop_min_w)


    # size of all signs
    sign_min = torch.min(torch.min(targets[:, 4]), torch.min(targets[:, 5])) * w    
    sign_max = torch.max(torch.max(targets[:, 4]), torch.max(targets[:, 5])) * w 
    print("sign_min, sign_max = ", sign_min, sign_max)
    # crop  a small image from the original image 
    s_ratio = img_size / large_sign  #  1000 / 100  = 10 
    l_ratio = img_size / small_sign  #  1000 / 10   = 100 
    print("s_ratio,l_ratio = ", s_ratio, l_ratio)
    # crop should between the two value below
 
    min_crop = max(sign_max * s_ratio, crop_min_size)
    max_crop = min(sign_min * l_ratio, w)
    if min_crop < max_crop:
        print("min_crop, max_crop = ", min_crop, max_crop)
        crop_w = random.uniform(min_crop, max_crop)
        # x1 = 
    
    # return img, (x1, y1, w, h), 
   # ###############################

   

def random_resize_ratio(img, targets, small_size, large_size):  # yzn
    # print(type(img))
    # print(type(targets))
    # print("img.size() = ", img.size())
    # print(targets.size())
    # print(targets)
    #  augment add by yzn 
    #  if all targets size > 0.5, this image should be resize to normal 
    # sign size
    min_wh = torch.min(torch.min(targets[:, 4]), torch.min(targets[:, 5]))  #  
    max_wh = torch.max(torch.max(targets[:, 4]), torch.max(targets[:, 5]))
    min_size = min_wh * img.shape[1]  # all objects size in this picture 
    max_size = max_wh * img.shape[1]  # 
    min_resize_sign_rat = small_size / min_size
    

    max_crop_rat = 1 / torch.max( torch.max(targets[:, 2]) + torch.max(targets[:, 4]), 
        torch.max(targets[:, 3]) + torch.max(targets[:, 5]))

    max_resize_sign_rat = torch.min(large_size / max_size,  max_crop_rat)

    # print("min and max = ", min_resize_sign_rat, max_resize_sign_rat)

    if min_resize_sign_rat + 0.2 < max_resize_sign_rat:
        # print("min_resize_sign_rat + 0.2, max_resize_sign_rat = ", min_resize_sign_rat + 0.2, max_resize_sign_rat)
        resize_rat = random.uniform(min_resize_sign_rat + 0.2, max_resize_sign_rat)
    else:
        resize_rat = 1
    # print("resize_rat = ", resize_rat)
    return resize_rat # resize_rat

 
def resize_to_smaller_bigger(img, targets, img_size, small_size=15, large_size=250):  # yzn 
    c, h, w = img.shape
    resize_rat = random_resize_ratio(img, targets, small_size=small_size, large_size=large_size)

    img, targets = crop_pad_to_size(img, targets, resize_rat, pad_value=0) 

    img = F.interpolate(img.unsqueeze(0), size=img_size, mode="nearest").squeeze(0)

    return img, targets



def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):  # not being used (yzn)
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img


    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=1216, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images_jpg", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 0 * 32
        self.max_size = self.img_size + 0 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))  #  yolo 
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

       
        img, targets = resize_to_smaller_bigger(img, targets, self.img_size, small_size=15, large_size=200)  # img is 1360 x 1360
        
        # show  image 
        # show_img_target(img, targets)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        img, targets = resize_to_smaller_bigger(img, targets, self.img_size, small_size=15, large_size=200)  # img is 1360 x 1360
        
        # ##########
        # show  image
        # ##########
        # show_img_target(img, targets)

        return img_path, img, targets



    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
      
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
