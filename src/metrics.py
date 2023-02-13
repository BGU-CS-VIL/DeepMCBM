####
# input:  background estimation directoty
#         input image directory
#         ground truth directory
# output: foreground estimation directory
#         write metrics and arguments to log file
####

import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import utils
import numpy as np
from utils import calc_F_measure, calc_TPR_FPR
from pathlib import Path
import pandas as pd
from metrics_roc import calc_roc
from natsort import natsorted


def image_list(dir_name):
    image_list = os.listdir(dir_name)
    image_list_paths = []
    for image_name in image_list:
        if not image_name.endswith((".avi", ".mp4", ".txt")):
            image_path = os.path.join(dir_name, image_name)
            image_list_paths.append(image_path)
    image_list_paths.sort()
    return image_list_paths


def calc_metrics(mse_dir, gt_dir, gt_shape, mse_shape, fg_threshold,CDNet = False):
    mse_list = image_list(mse_dir)
    gt_list = image_list(gt_dir)
    # get shape
    gt = ToTensor()(Image.open(gt_list[0]).convert('RGB'))
    gt_shape = gt.shape[1:]


    assert len(mse_list) == len(gt_list), "bg,input and gt lists must have the same length"

    metrics = {'F_measure': [], 'precision': [], 'recall': [], "TPR": [], "FPR": []}
    resize_to_gt = Resize(gt_shape)
    resize_to_bg = Resize(mse_shape)
    max_mse = 0
    for mse_, gt_ in zip(mse_list, gt_list):
        mse = Image.open(mse_).convert('RGB')
        mse = ToTensor()(mse)
        gt = Image.open(gt_)
        gt = ToTensor()(gt)

        mse = resize_to_gt(mse)  # resize to source shape
        if len(mse.shape) == 3 :
            mse = mse[0]
            mse = mse[None,:,:]
        
        if CDNet:
            # for CDNet dataset
            # 0 : Static
            # 50 : Hard shadow
            # 85 : Outside region of interest
            # 170 : Unknown motion (usually around moving objects, due to semi-transparency and motion blur)
            # 255 : Motion
            out_of_ROI_val_1 = 85/255
            out_of_ROI_val_2 = 170/255
            eps = 1e-5
            a = torch.abs(gt - out_of_ROI_val_1)
            ROI_idx = torch.where(torch.abs(gt - out_of_ROI_val_1) > eps,True,False) 
            gt = gt[ROI_idx]
            mse = mse[ROI_idx]
            ROI_idx = torch.where(torch.abs(gt - out_of_ROI_val_2) > eps,True,False) 
            gt = gt[ROI_idx]
            mse = mse[ROI_idx]
            motion_val = 255/255 
            gt = torch.where(gt>1-eps,1.0,0.0)

        if len(gt)>0:
            if mse.max() > max_mse:
                max_mse = mse.max()
            gt = torch.where(gt > 0, 1.0, 0.0)
            fg_est = torch.where(mse > fg_threshold, 1.0, 0.0)
            metrics_vals = utils.calc_F_measure(
                fg_est.cpu().numpy(), gt.cpu().numpy())

            for key, val in zip(metrics.keys(), metrics_vals):
                metrics[key].append(val)

    print("------Max MSE: ", max_mse)
    return metrics


def generate_mse(video_path, bg_path, MSE_dir):
    # get list of file names, sorted
    frames_list = image_list(video_path)
    bg_list = image_list(bg_path)
    assert len(frames_list) == len(
        bg_list), f"background and frame lists must have the same length, got {len(frames_list)} and {len(bg_list)}"
    Path(MSE_dir).mkdir(parents=True, exist_ok=True)


    # sort frames by name ["frame_1.png", "frame_2.png", ...]
    bg_list = natsorted(bg_list)
    frames_list = natsorted(frames_list)
    
    mse_list = []
    for i, (bg, frame) in enumerate(zip(bg_list, frames_list)):
        frame_name = f"frame{i:06}.png"
        frame_path = os.path.join(MSE_dir, frame_name)

        bg = Image.open(bg)
        mode = bg.mode
        frame = Image.open(frame)
        frame = frame.convert(mode)
        bg = ToTensor()(bg)
        frame = ToTensor()(frame)
        resize_to_bg = Resize((bg.shape[1], bg.shape[2]))
        frame = resize_to_bg(frame)
        frame_mse = torch.mean((frame - bg) ** 2, dim=0)
        mse_list.append(frame_mse)
        utils.save_image(frame_path, frame_mse)

def calc_metric_and_MSE(video_path, bg_path, gt_path, mse_path, args, method="", overwrite=True):
    """
    Calculates F1, precision and recall measure per-frame (mean/std)
     and entire video (mean/std). 
        Calculates MSE per frame and saves it under a new dir
    run: Log metrics to Neptune 
    Args - all paths should be full paths:
        video_path: path to video
        bg_path: path to background estimation
        gt_path: path to ground truth labels. 
        mse_path: path to MSE. Will be created if it doesn't exist
        run: Log metrics to Neptune 
        args: arguments
    Returns:
        logs metrics to neptune
    """

    # check if MSE dir exist
    # if not,  generate MSE between background and frame and save
    # mignt need to resize frame to background``
    # if exist we can continue, otherwise we'll calc MSE and save at dir
    generate_mse(video_path, bg_path, mse_path)
    FPR, TPR, AUC = calc_roc(mse_dir=mse_path, gt_dir=gt_path, gt_shape=args.source_shape,
                        mse_shape=args.mask_shape,CDNet=args.CDNet)

    FPR = FPR.tolist()
    TPR =  TPR.tolist()
    print("TPR len:", len(TPR))
    print("FPR len:", len(FPR))
    print("AUC score:", AUC)
    return FPR, TPR
