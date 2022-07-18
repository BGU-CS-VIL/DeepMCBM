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
from ConvAE import ConvAE
import utils
from utils import update_special_args
from image_dataset import ImageDataset
from BMN import BMN
from GM_loss import GM
import args as ARGS
import metrics
import STN_Homo



def set_optimizer_and_scheduler(bmn,args,run):
    optimizer = torch.optim.Adam(
    bmn.parameters(), lr=args.AE_lr, weight_decay=args.AE_weight_decay)
    run['config/optimizer'] = type(optimizer).__name__
    if args.AE_schedul_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.AE_schedul_step, gamma=args.AE_schedul_gamma)
    elif args.AE_schedul_type == 'Mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.AE_schedul_milestones, gamma=args.AE_schedul_gamma)
    run['config/scheduler'] = type(scheduler).__name__
    return optimizer,scheduler

def log(run,epoch,epoch_loss,lr,sample_image,bmn,log_interval):
    run["training/loss"].log(epoch_loss)
    run["training/epoch"].log(epoch)
    run["training/lr"].log(lr)
    print("epoch:",epoch)
    print("loss:",epoch_loss)
    print("lr :",lr)
    if epoch%log_interval == 0:
        log_frame_reconstruction(bmn,sample_image,epoch,run)

def log_frame_reconstruction(bmn,image,epoch,run):
    with torch.no_grad():
        bmn.eval()
        reconstruction, AE_output, warped_image, transform,warped_mean = bmn(image)
        bmn.train()
    utils.log_image(reconstruction, 
                    f"epoch {epoch}",
                    run, 
                    f'training/reconstruction')
    utils.log_image(AE_output, "epoch = " +
                            str(epoch), run, 'training/AE_output')
    utils.log_image(warped_mean, "epoch = " +
                            str(epoch), run, 'training/STN_output')

def main(args):
    ## PARAMS and SETTINGS 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    ## LOGGER
    NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOWQzYWJiNy0wNDk5LTQxZDctOTlmMi1kN2JmYjJmOWViZTEifQ=="
    run = neptune.init(project='vil/background-AE',
                    api_token=NEPTUNE_API_TOKEN,
                    source_files=['*.py'],
                    tags=args.tags)
    
    # args = update_special_args(args, args.special_args_path)
    run['config/params'] = vars(args)
    ckpt_name = args.dir+"_"+ utils.fetch_Neptune_run_id(run)
    ckpt_path = os.path.join(args.BMN_ckpt_dir,ckpt_name)

    ## DATA 
    train_transforms = transforms.Compose([
        transforms.Resize(size=args.mask_shape),
        transforms.ToTensor(),
    ])
    root = os.path.join(args.parent_dir, args.dir, "frames")
    print(f"working on dataset in:{root}")
    run['config/dataset/path'] = os.path.join(args.parent_dir, args.dir)

    train_loader = DataLoader(
        ImageDataset(root=root,
                    transform=train_transforms),
        batch_size=args.AE_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False)

    first_frame = train_loader.dataset.__getitem__(0)[None,:,:,:].to(device)
    utils.log_image(first_frame, "first_frame", run, 'training/first_frame')

    ## MODEL
    # load STN
    if args.TG == 'Affine':
        stn = STN.STN_block(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                        use_homography=args.homography, use_cpab=args.cpab,
                        zero_boundary=args.cpab_zero_boundary).to(device)
    
    if args.TG == 'Homo':
        stn = STN_Homo.STN_Homo(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                use_homography=args.homography, use_cpab=args.cpab,
                zero_boundary=args.cpab_zero_boundary).to(device)
                    
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
    utils.log_image(bmn.moments[0],"robust mean",run,"training/robust_mean")

    ## LOSS
    if args.AE_loss_type == 'MSE':
        reconstruction_loss = nn.MSELoss()
    elif args.AE_loss_type == 'SL1':
        reconstruction_loss = nn.SmoothL1Loss(beta=args.AE_beta)
    elif args.AE_loss_type == 'GM':
        reconstruction_loss = GM(sigma=args.sigma)


    ## OPTIMIZER and SCHEDULER 
    optimizer,scheduler = set_optimizer_and_scheduler(bmn,args,run)

    ## TRAIN
    # init 
    size = len(train_loader.dataset)
    first_frame = train_loader.dataset.__getitem__(0)[None,:,:,:].to(device)
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
        # End of epoch routine 
        epoch_loss = running_loss/size
        log(run,epoch,epoch_loss,scheduler.get_last_lr(),first_frame,bmn,args.log_interval)
        scheduler.step()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            utils.save_model(ckpt_path+"_best.ckpt",bmn,optimizer,scheduler)

    # End of training
    utils.save_model(ckpt_path+"_last.ckpt",bmn,optimizer,scheduler)
    run.stop()
    return ckpt_name

if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)