import torch 
from PIL import Image
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, info_file, names, transform=None):
        f_info = open(info_file)
        f_names = open(names)
        info_list = f_info.readlines()
        names_list = f_names.readlines()
        self.paths = [i.split(" ")[0] for i in info_list ]
        self.labels = [i.split(" ")[1] for i in info_list ]
        self.names = [name.replace("\n", "") for name in names_list]
        targets = [[self.names.index(lable)] for lable in self.labels]
        self.targets = torch.LongTensor(targets)
        self.transform = transform
      

    def __len__(self):
        return  len(self.paths)


    def __getitem__(self, index):
        img_path = self.paths[index % len(self.paths)].rstrip() 
        img = Image.open(img_path)  # .convert('L')   # .convert('L')  # .convert('L')   # .resize((38, 28))  #  to  gray and  resize
        # # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img))
        # img = Image.fromarray(img)
        return img_path, self.transform(img), self.targets[index]
  
