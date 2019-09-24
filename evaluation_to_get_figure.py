from __future__ import division

from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter



import os
os.environ['CUDA_VISIBLE_DEVICES']='6'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    # parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/ALL_DATA.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/ALL_DATA.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--n_cpu", ype=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1216, help="size of each image dimension")
    # parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    # parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()


    writer = SummaryWriter(log_dir="log_val_figure_dir_class_1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    
    valid_path = data_config["test"]  # test  valid
    class_names = load_classes(data_config["names"])

    model = Darknet(opt.model_def) #  add by yzn
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
    model.to(device)

    

    weights_dir = "checkpoints"
    weights = os.listdir(weights_dir)

    for i in range(81, len(weights), 3):
        
        path = os.path.join(weights_dir, weights[i])

        model.load_state_dict(torch.load(path))

        print("\n---- Evaluating Model ---- %d", i)
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=1,  #  default set 8, will :RuntimeError: CUDA out of memory. 
            # Tried to allocate 95.12 MiB (GPU 0; 7.93 GiB total capacity; 7.19 GiB already allocated; 
            # 94.56 MiB free; 95.97 MiB cached
        )
        evaluation_metrics = {
            "val_precision": precision.mean(),
            "val_recall": recall.mean(),
            "val_mAP": AP.mean(),
            "val_f1": f1.mean(),
        }
        writer.add_scalars('data/scalar_metrics', evaluation_metrics, i) # logger

        # print("class_names =", class_names)
        # print("ap_class = ", ap_class)  #  [ 0  2  4  7  9 12 15 17 18 20 23]

        ap_table = {class_names[c]:AP[i] for i, c in enumerate(ap_class)}
        print("ap_table= ", ap_table)
        writer.add_scalars('data/scalar_ap_class', ap_table, i) # logger



        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):

            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]


        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
