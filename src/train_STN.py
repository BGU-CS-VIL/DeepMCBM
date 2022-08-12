import torch
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Pad, Resize, Compose, ToTensor
import neptune.new as neptune
import matplotlib.pyplot as plt
from torchsummary import summary
from utils import update_special_args

import utils
import Loss
import STN
import STN_Homo
from image_dataset import ImageDataset
import args as ARGS

def set_optimizer_and_scheduler(model,args):
    if args.STN_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = args.STN_lr,
                                    momentum = args.STN_momentum,
                                    weight_decay = args.STN_weight_decay)
    elif args.STN_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr = args.STN_lr,    
                                    weight_decay = args.STN_weight_decay)
    if args.STN_schedul_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size = args.STN_schedul_step, 
                                                    gamma = args.STN_schedul_gamma)
    elif args.STN_schedul_type == 'Mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones = args.STN_schedul_milestones, 
                                                        gamma = args.STN_schedul_gamma)
    return optimizer,scheduler

def log(epoch,epoch_loss,lr):
    print("epoch:",epoch)
    print("loss:",epoch_loss)
    print("lr :",lr)

    
def main(args,**kwargs):
    ## Params and Settings 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    ## Logger
    ckpt_name = args.dir+"_"+args.log_name
    ckpt_path = os.path.join(args.STN_ckpt_dir,ckpt_name)

    ## Data 
    image_path = os.path.join(args.parent_dir, args.dir, "frames")
    print(f"working on dataset in:{image_path}")
    transform = Compose([
        Resize(args.mask_shape),
        ToTensor()
    ])
    dataset = ImageDataset(image_path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=args.STN_batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=4)
    ## Model
    if args.TG == 'Affine':
        stn = STN.STN_block(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                            use_homography=args.homography).to(device)
    if args.TG == 'Homo':
        stn = STN_Homo.STN_Homo(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                        use_homography=args.homography).to(device)
        if args.load_Affine:
            stn.Load_Backbone_and_AffineHead(args)

    ## Loss
    panorama_shape = (args.channels, args.mask_shape[0]+args.pad[0], args.mask_shape[1]+args.pad[1])
    alignment_loss = Loss.Alignment_Loss(panorama_shape,
                                        args.memory, 
                                        SmoothL1Loss_beta=args.beta).to(device)
    alignment_loss.init_loss(dataloader,stn)
    # TODO : save init alignment image
    
    ## Optimizer and Scheduler
    optimizer,scheduler = set_optimizer_and_scheduler(stn,args) 
    
    ## Training 
    # Init 
    size = len(dataloader.dataset)
    min_loss = float('inf')
    stn.train()

    # Training Loop 
    for epoch in range(args.STN_total_epochs):
        running_loss = 0 
        for batch in dataloader:
            optimizer.zero_grad()
            warped_image, warped_mask, transform = stn(batch.to(device))
            loss = alignment_loss(warped_image, warped_mask)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        # End of Epoch
        epoch_loss = running_loss/size
        log(epoch,epoch_loss,scheduler.get_last_lr())
        scheduler.step()
        alignment_loss.step()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            utils.save_model(ckpt_path+"_STN_best.ckpt",stn,optimizer,scheduler)

    # End of Training
    utils.save_model(ckpt_path+"_STN_last.ckpt",stn,optimizer,scheduler)
    return ckpt_name
    
if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)