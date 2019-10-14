from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import datasets

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
 

from utils.utils import rescale_boxes
from utils.datasets import pad_to_square, resize

from tqdm import tqdm
 


from ALL_sign_data.model import Lenet5, my_resnt18, FashionCNN
from ALL_sign_data.resnet import ResNet18

import json


os.environ['CUDA_VISIBLE_DEVICES']='3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sign_classes = 115
# classes_weights_path = "ALL_sign_data/model_acc_90__epoch_4.pt"

classes_weights_path = "ALL_sign_data/checkpoints_4/model_acc_94__calss_115_epoch_14.pt"

# os.makedirs("output", exist_ok=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/changshu_18_during", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/ALL_DATA.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_13.pth", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_33.pth", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/ALL_DATA.names", help="path to class label file")
    parser.add_argument("--class_path", type=str, default="ALL_sign_data/ALL_data_in_2_train/names.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1216, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)


    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    # dataloader = DataLoader(
    #     ImageFolder(opt.image_folder, img_size=opt.img_size),
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.n_cpu,
    # )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    # prev_time = time.time()

    # to  class
    model_class = FashionCNN(sign_classes)
    # model_class = ResNet18(sign_classes)
    

    model_class.load_state_dict(torch.load(classes_weights_path))
    model_class.to(device)
    model_class.eval()
    # to  class


    crop_dirs = [
    # "/headless/Desktop/yzn_file/DataSetsH/CCTSDB/CCTSDB/test_data/img/",  # 400
    # "/headless/Desktop/yzn_file/DataSetsH/WIDER_FACE/WIDER_test/images/0--Parade/",  # 575
    # "/headless/Desktop/yzn_file/DataSetsH/baiduApollo/demo_data/trainsets/images/",  # 100
    # "/headless/Desktop/yzn_file/DataSetsH/baiduApollo/demo_data/testsets/images/",  # 100
    # "/headless/Desktop/yzn_file/DataSetsH/CCTSDB/CCTSDB/images/",  # 15724
    # "/headless/Desktop/yzn_file/DataSetsH/CCTSDB/CCTSDB/test_data/img/",  # 400
    # "/headless/Desktop/yzn_file/DataSetsH/COCO/test2014/",  # 40775
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficPanelDatabase/data_A_TraffficPanelDatabaseA/",  # 1524
    
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficPanelDatabase/data_B_TrafficPanelDatabaseB/",  # 976
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficSignDetectionDatabase/test_data_TsignDetTestDatabase/",
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficSignDetectionDatabase/train_data_TsignDetTrainDatabase/",
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficSignRecogntionDatabase/test_data_TSRD-Test/",
    # "/headless/Desktop/yzn_file/DataSetsH/CTSD_zidonghua/TrafficSignRecogntionDatabase/train-data_tsrd-train/",
    # "/headless/Desktop/yzn_file/DataSetsH/DFGTSD/JPEGImages/JPEGImages/",

    # "/headless/Desktop/yzn_file/DataSetsH/gangjinDataSet/train_dataset/", 
    # "/headless/Desktop/yzn_file/DataSetsH/gangjinDataSet/test_dataset/",  
    # "/headless/Desktop/yzn_file/DataSetsH/GTSDB/FullIJCNN2013",
    # "/headless/Desktop/yzn_file/DataSetsH/httpcvrr.ucsd.eduLISAlisa-traffic-sign-dataset.html/signDatabasePublicFramesOnly/aiua120306-1/frameAnnotations-DataLog02142012_003_external_camera.avi_annotations/",
    # "/headless/Desktop/yzn_file/DataSetsH/Tinghua100K/data_all/train/",
    # "/headless/Desktop/yzn_file/DataSetsH/Tinghua100K/data_all/test/",
    # "/headless/Desktop/yzn_file/DataSetsH/Tinghua100K/data_all/other/",
    # "/headless/Desktop/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2012/JPEGImages/",
    # "/headless/Desktop/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/",
    
    
    # "data/changshu_17_before", 
    # "data/changshu_18_before",
    # "data/changshu_18_during",
    
    # "data/samples",
    # "data/samples_2",
    # "data/samples_changsha"

    # "/headless/Desktop/yzn_file/DataSetsH/DFGTSD/JPEGImages/JPEGImages/", 
    # "/headless/Desktop/yzn_file/DataSetsH/Tinghua100K/data_all/train_just/",
    "image_for_detect/Tinghua100K"
                ]

    for dir_ in crop_dirs:
        train_results = {"imgs" : {}}

        opt.image_folder = dir_

        # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        names = os.listdir(opt.image_folder)
        nums = 0
        for name in tqdm(names[:]):
            img_id = name.split(".")[0]

            print("data is: ", dir_)
            print("name is: ", name)
            img_path = os.path.join(opt.image_folder, name)
            print("img_path:", img_path)
            if not os.path.isfile(img_path):
                continue

            if not (img_path.endswith(".jpg")  or img_path.endswith(".png") or img_path.endswith(".ppm")  ):  #  
                continue
            # if 


            # Extract image as PyTorch tensor
            img = torchvision.transforms.ToTensor()(Image.open(img_path).convert(mode="RGB"))
          
            input_imgs, _ = pad_to_square(img, 0)
            # Resize
            input_imgs = resize(input_imgs, opt.img_size).unsqueeze(0)
     

     


            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs.to(device))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]


            # Log progress
            # current_time = time.time()
            # inference_time = datetime.timedelta(seconds=current_time - prev_time)
            # prev_time = current_time
            # print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections

            # print("detections = ", detections)
            # imgs.extend(img_paths)
            # img_detections.extend(detections)
            if  detections is not None:  #  one image
                objects = []  #  save the results of a image detection
                detections = rescale_boxes(detections, opt.img_size, img.shape[1:])
                
                unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                # bbox_colors = random.sample(colors, n_cls_preds)
                # plt.figure()
                fig, ax = plt.subplots()
                img_copy =Image.open(img_path) 
                # ax.imshow(img_copy)
                j = 0
                
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections): #  one object in a image
                
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    min_sign_size = 10

                    if box_w >= min_sign_size and box_h >= min_sign_size:
                        
                        crop_sign_org = img_copy.crop((x1, y1, x2, y2)).convert(mode="RGB")
            

                        # #### to class  ###############
                        test_transform = torchvision.transforms.Compose([ 
                            torchvision.transforms.Resize((28, 28), interpolation=2),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                            ])

                        crop_sign_input = test_transform(crop_sign_org).unsqueeze(0)
                        # input_img = torch.autograd.Variable(input_img)

                        # print("input_img = ", input_img.size())
                        with torch.no_grad():
                            pred_class = model_class(crop_sign_input.to(device))
                        sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
                        # #### to class  ###############
                        cls_pred = sign_type

                        print("cls_pred_type = ", classes[int(cls_pred)])
                        # #############
                        # save crop image
                        # #############                        
                        if classes[int(cls_pred)] != "zo":
                            # save  crop image #############
                            save_dir = "img_crop_2_classification_Tinghua_weights_11/" + classes[int(cls_pred)]
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            name = dir_.split("/")[-2] + "_" + dir_.split("/")[-1] + str(int(random.random() * 100000000))
                            print("save path:", save_dir, str(name) + ".jpg")
                            #  save crop sign
                            crop_sign_org.save(os.path.join(save_dir, str(name) + ".jpg"))
                       
                        # #####
                        #
                        # draw image 
                        #
                        # #####
                        if True and classes[int(cls_pred)] != "zo":     
                            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            color = "r"
                            # Create a Rectangle patch
                            # plt.imshow()
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                            # Add the bbox to the plot
                            ax.add_patch(bbox)
                            # Add label
                            plt.text(
                                x1,
                                y2 + 50,
                                s=classes[int(cls_pred)],
                                color="white",
                                verticalalignment="top",
                                bbox={"color": color, "pad": 0},
                            )
                            

                            pad_sign_path_png = "ALL_sign_data/pad-all/" + classes[int(cls_pred)] + ".png"
                            pad_sign_path_jpg = "ALL_sign_data/pad-all/" + classes[int(cls_pred)] + ".jpg"
                            if  os.path.isfile(pad_sign_path_png):
                                pad_sign = Image.open(pad_sign_path_png)
                            elif os.path.isfile(pad_sign_path_jpg):
                                pad_sign = Image.open(pad_sign_path_jpg)
                            else:
                                pad_sign = Image.new("RGB", (100, 100), (255, 255, 255))

                            img_copy.paste(crop_sign_org.resize((100, 100)), (0, j * 100) )
                            img_copy.paste(pad_sign.resize((100, 100)), (100, j * 100) )
                            j += 1
                            
                            #  save predict results to a json file: my_train_results.json
                            objects.append({'category': classes[int(cls_pred)], 'score': 848.0, 'bbox': {'xmin': x1, 'ymin': y1, 'ymax': y2, 'xmax': x2}})



                # Save generated image with detections
                nums += 1
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                


                # # Save generated image with detections
                # plt.axis("off")
                # plt.gca().xaxis.set_major_locator(NullLocator())
                # plt.gca().yaxis.set_major_locator(NullLocator())
                # filename = path.split("/")[-1].split(".")[0]
                # plt.savefig(f"output/{nums}.png", bbox_inches="tight", pad_inches=0.0,)
                # plt.close()                



                ax.imshow(img_copy)

                # plt.ion()
                # plt.pause(0.5)
                # plt.close()
                try:
                    dir__ = dir_.split("/")[-1]
                    plt.savefig(f"output/{dir__ + str(nums).zfill(5)}.png", bbox_inches="tight", pad_inches=0.0,)
                except:
                    continue
                # plt.show()
            train_results["imgs"][img_id] = {"objects": objects}
            # print("train_results = ", train_results)

    file_name = "results/Tinghua100K_result.json"
    with open(file_name, "w") as file_object:
        json.dump(train_results, file_object)
 






        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, dpi=400)
        # plt.close()



    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")



    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    #     # print("(%d) Image: '%s'" % (img_i, path))

    #     # Create plot
    #     img = np.array(Image.open(path))
    #     print("img.shape = ", img.shape) # ( 2048 2048 3 )

    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)
        
    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
        #     detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        #     unique_labels = detections[:, -1].cpu().unique()
        #     n_cls_preds = len(unique_labels)
        #     bbox_colors = random.sample(colors, n_cls_preds)
        #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        #         box_w = x2 - x1
        #         box_h = y2 - y1

        #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        #         color = "r"
        #         # Create a Rectangle patch
        #         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
        #         # Add the bbox to the plot
        #         ax.add_patch(bbox)
        #         # Add label
        #         plt.text(
        #             x1,
        #             y2 + 50,
        #             s=classes[int(cls_pred)],
        #             color="white",
        #             verticalalignment="top",
        #             bbox={"color": color, "pad": 0},
        #         )

        # # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, dpi=400)
        # plt.close()


