import numpy as np 
import torch
from sklearn.metrics import roc_curve, auc
import os
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import utils
from utils import calc_F_measure, calc_TPR_FPR
from pathlib import Path
import pandas as pd
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

def calc_roc(mse_dir, gt_dir, gt_shape, mse_shape,CDNet = False):
    mse_list = image_list(mse_dir)
    gt_list = image_list(gt_dir)
    # get shape
    gt = ToTensor()(Image.open(gt_list[0]).convert('RGB'))
    gt_shape = gt.shape[1:]


    assert len(mse_list) == len(gt_list), "bg,input and gt lists must have the same length"

    metrics = {'F_measure': [], 'precision': [], 'recall': [], "TPR": [], "FPR": []}
    resize_to_gt = Resize(gt_shape)
    resize_to_bg = Resize(mse_shape)
    mse_arr = []
    gt_arr = []
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
            gt = torch.where(gt > 0, 1.0, 0.0)
            mse_arr.append(mse)
            gt_arr.append(gt)
        
    # scale across entire video - shape (N_frames, H, W)
    mse_vid = np.concatenate(mse_arr, axis=0)
    gt_vid = np.concatenate(gt_arr, axis=0)
    mse_scaled = (mse_vid - np.min(mse_vid)) / (np.max(mse_vid)- np.min(mse_vid))
    
    # compute roc for batch
    mse_vec = mse_scaled.ravel()
    gt_vec = gt_vid.ravel()
    
    debug = False
    if debug: 
        print("mse_vec has nan:",np.any(np.isnan(mse_vec)))
        print("gt_vec has nan:",np.any(np.isnan(gt_vec)))
        print("mse_vec has inf:",np.any(np.isinf(mse_vec)))
        print("gt_vec has inf:",np.any(np.isinf(gt_vec)))
        print("mse_vec max:",mse_vec.max())

    FPR, TPR, _ = roc_curve(gt_vec.astype(int), np.round_(mse_vec, 4), drop_intermediate=True) # drop_intermediate and round for mmeory issue 
    AUC = auc(FPR, TPR)
    return FPR, TPR, AUC
