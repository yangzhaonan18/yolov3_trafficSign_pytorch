import os
import cv2
import numpy as np
from tqdm import tqdm


path = 'output'  #  save images
filelist = os.listdir(path)

fps = 24 #ÊÓÆµÃ¿Ãë24Ö¡
size = (1278, 1247) #ÐèÒª×ªÎªÊÓÆµµÄÍ¼Æ¬µÄ³ß´ç
#¿ÉÒÔÊ¹ÓÃcv2.resize()½øÐÐÐÞ¸Ä
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("VideoTe3st1.avi", fourcc, fps, size)
#ÊÓÆµ±£´æÔÚµ±Ç°Ä¿Â¼ÏÂ

for item in tqdm(filelist):
    if item.endswith('.png'): 
    #ÕÒµ½Â·¾¶ÖÐËùÓÐºó×ºÃûÎª.pngµÄÎÄ¼þ£¬¿ÉÒÔ¸ü»»Îª.jpg»òÆäËü

        img_path = os.path.join(path, item)
        # print(img_path)
        img = cv2.imread(img_path)
        print(img.shape)
        print(img)
        cv2.imshow("asdf", img)
        video.write(img)

video.release()
cv2.destroyAllWindows()