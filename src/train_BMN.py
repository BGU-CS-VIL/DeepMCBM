import torch
import os
from torch.utils.data import DataLoader, dataloader
from torchvision.transforms import transforms
import neptune.new as neptune
from torchsummary import summary
import torch.nn as nn
import STN
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from autoencoder import AutoEncoder
# from ConvAE import ConvAE
import utils
from utils import update_special_args
from image_dataset import ImageDataset
from BMN import BMN
from GM_loss import GM
import args as ARGS
import metrics
import STN_Homo



def set_optimizer_and_scheduler(bmn,args):
    optimizer = torch.optim.Adam(
    bmn.parameters(), lr=args.AE_lr, weight_decay=args.AE_weight_decay)
    if args.AE_schedul_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.AE_schedul_step, gamma=args.AE_schedul_gamma)
    elif args.AE_schedul_type == 'Mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.AE_schedul_milestones, gamma=args.AE_schedul_gamma)
    return optimizer,scheduler

def log(epoch,epoch_loss,lr):
    print("epoch:",epoch)
    print("loss:",epoch_loss)
    print("lr :",lr)
    
def main(args,**kwargs):
    ## PARAMS and SETTINGS 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")
    
    # args = update_special_args(args, args.special_args_path)
    ckpt_name = args.dir+"_"+ args.log_name
    ckpt_path = os.path.join(args.BMN_ckpt_dir,ckpt_name)

    ## DATA 
    train_transforms = transforms.Compose([
        transforms.Resize(size=args.mask_shape),
        transforms.ToTensor(),
    ])
    root = os.path.join(args.parent_dir, args.dir, "frames")
    print(f"working on dataset in:{root}")

    train_loader = DataLoader(
        ImageDataset(root=root,
                    transform=train_transforms),
        batch_size=args.AE_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False)

    first_frame = train_loader.dataset.__getitem__(0)[None,:,:,:].to(device)

    ## MODEL
    # load STN
    if args.TG == 'Affine':
        stn = STN.STN_block(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                        use_homography=args.homography).to(device)
    
    if args.TG == 'Homo':
        stn = STN_Homo.STN_Homo(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                use_homography=args.homography).to(device)
                    
    checkpoint_path = os.path.join(args.STN_ckpt_dir, args.STN_ckpt)
    checkpoint = torch.load(checkpoint_path)
    stn.load_state_dict(checkpoint['state_dict'])
    stn.to(device)
    print("STN was loaded from: ", checkpoint_path)
    # build AE 
    in_chan = args.channels*(args.moments_num + 1)
    ae = AutoEncoder(C=args.C,
                        M=args.M,
                        in_chan=in_chan,
                        out_chan=args.channels,
                        input_shape=args.mask_shape,
                        code_size=args.code_size).to(device)
    # build BMN with the loaded STN and the created AE 
    bmn = BMN(args.moments_num, stn, ae, use_theta_embedding=args.theta_embedding,
              cond_decoder=args.cond_decoder).to(device)
    # init BMN moments 
    bmn.init_moments(train_loader,args.trim_percentage)
    # TODO : save robust mean 

    ## LOSS
    if args.AE_loss_type == 'MSE':
        reconstruction_loss = nn.MSELoss()
    elif args.AE_loss_type == 'SL1':
        reconstruction_loss = nn.SmoothL1Loss(beta=args.AE_beta)
    elif args.AE_loss_type == 'GM':
        reconstruction_loss = GM(sigma=args.sigma)

    ## OPTIMIZER and SCHEDULER 
    optimizer,scheduler = set_optimizer_and_scheduler(bmn,args)

    ## TRAIN
    # init 
    size = len(train_loader.dataset)
    min_loss = float('inf')
    bmn.train()
    # training loop 
    for epoch in range(args.AE_total_epochs):
        running_loss = 0.0 
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs, AE_output, warped_input_image, transform,warped_mean = bmn(batch)
            loss = reconstruction_loss(outputs,batch)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        # end of epoch routine 
        epoch_loss = running_loss/size
        log(epoch,epoch_loss,scheduler.get_last_lr())
        scheduler.step()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            utils.save_model(ckpt_path+"_BMN_best.ckpt",bmn,optimizer,scheduler)
    # end of training
    utils.save_model(ckpt_path+"_BMN_last.ckpt",bmn,optimizer,scheduler)
    return ckpt_name

if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)