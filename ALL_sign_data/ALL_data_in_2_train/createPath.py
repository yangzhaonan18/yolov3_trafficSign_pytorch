import os
from tqdm import tqdm
from PIL import Image

dirs = os.listdir("all_crop_data")
dirs = sorted(dirs)
f_types = open("names.txt", "w")
for dr in dirs:
    f_types.write(dr+"\n")
f_types.close()


f = open("info.txt", "w")
# root = os.getcwd()
# root = './all_crop_data'
root = "/headless/Desktop/yzn_file/code/PyTorch-YOLOv3-Tinghua100K/ALL_sign_data/ALL_data_in_2_train/all_crop_data"

for dr in tqdm(dirs):
    names = os.listdir("all_crop_data/" + dr)
    for name in names:
        img_path = root+"/"+ dr + "/"+ name 
        try:
            Image.open(img_path)
            f.write(img_path + " " + dr + " " + name.split("_")[0] + "\n")
        except:
            print("bug")
            # os.remove(img_path)
            print("img_path = ", img_path)
   
f.close()

