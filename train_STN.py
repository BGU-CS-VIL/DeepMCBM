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

def set_optimizer_and_scheduler(stn,args,run):
    if args.STN_optimizer == 'SGD':
        optimizer = torch.optim.SGD(stn.parameters(), 
                                    lr = args.STN_lr,
                                    momentum = args.STN_momentum,
                                    weight_decay = args.STN_weight_decay)
    elif args.STN_optimizer == 'Adam':
        optimizer = torch.optim.Adam(stn.parameters(), 
                                    lr = args.STN_lr,    
                                    weight_decay = args.STN_weight_decay)
    run['config/optimizer'] = type(optimizer).__name__
    if args.STN_schedul_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size = args.STN_schedul_step, 
                                                    gamma = args.STN_schedul_gamma)
    elif args.STN_schedul_type == 'Mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones = args.STN_schedul_milestones, 
                                                        gamma = args.STN_schedul_gamma)
    run['config/scheduler'] = type(scheduler).__name__

    return optimizer,scheduler


def log(run,epoch,epoch_loss,lr,alignment_image,first_frame,stn,log_interval):
    run["training/STN/loss"].log(epoch_loss)
    run["training/STN/epoch"].log(epoch)
    run["training/STN/lr"].log(lr)
    print("epoch:",epoch)
    print("loss:",epoch_loss)
    print("lr :",lr)
    if epoch%log_interval == 0:
        utils.log_image(alignment_image, f"epoch {epoch}",
                run, 'training/STN/alignment_image')
        log_frame_alignment(stn,first_frame,"first_frame_warped",epoch,run)
    
def log_frame_alignment(stn,image,title,epoch,run):
    with torch.no_grad():
        stn.eval()
        image_out,mask_out,theta_out = stn(image)
    stn.train()
    utils.log_image(image_out, 
                    f"epoch {epoch}",
                    run, 
                    f'training/STN/{title}')
    

def main(args,**kwargs):
    ## Params and Settings 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    ## LOGGER
    if 'run' in kwargs.keys():
        run = kwargs['run']
    else:
        # NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOWQzYWJiNy0wNDk5LTQxZDctOTlmMi1kN2JmYjJmOWViZTEifQ=="
        run = neptune.init(project=args.neptune_project,
                        api_token=args.neptune_api_token,
                        source_files=['*.py'],
                        tags=args.tags)
    
    run['config/params'] = vars(args)
    ckpt_name = args.dir+"_"+utils.fetch_Neptune_run_id(run)
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
    run['config/dataset/path'] = os.path.join(args.parent_dir, args.dir)

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
    utils.log_image(alignment_loss.mu, "init alignment",
                    run, 'training/STN/alignment_image')
    
    ## Optimizer and Scheduler
    optimizer,scheduler = set_optimizer_and_scheduler(stn,args,run) 

    
    ## Training 
    # Init 
    size = len(dataloader.dataset)
    first_frame = dataloader.dataset.__getitem__(0)[None,:,:,:].to(device)
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
        log(run,epoch,epoch_loss,scheduler.get_last_lr(),alignment_loss.mu,first_frame,stn,args.log_interval)
        scheduler.step()
        alignment_loss.step()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            utils.save_model(ckpt_path+"_STN_best.ckpt",stn,optimizer,scheduler)

    # End of Training
    utils.save_model(ckpt_path+"_STN_last.ckpt",stn,optimizer,scheduler)
    # run.stop()
    return ckpt_name
    
if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)