import torch
import os
from torch.utils.data import DataLoader, dataloader
from torchvision.transforms import transforms
import neptune.new as neptune
import STN
import STN_Homo
from tqdm import tqdm
import pathlib

from autoencoder import AutoEncoder
import utils
from utils import update_special_args
from image_dataset import ImageDataset
from BMN import BMN
import args as ARGS
import metrics
from metrics import calc_metric_and_MSE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_BMN(bmn,data_loader,exp_path):
    # create output directories
    print("saving results to:", exp_path)
    bg_est_path = os.path.join(exp_path, "background_estimation")
    pathlib.Path(bg_est_path).mkdir(parents=True, exist_ok=True)
    utils.save_image(os.path.join(exp_path, "panoramic_robust_mean.png"), bmn.moments[0])

    i = 0
    bmn.eval()
    for images in tqdm(data_loader, desc="predict background",total=len(data_loader)):
        images = images.to(device, dtype=torch.float32)
        background,_ = bmn.predict(images)
        for bg in background:
            name = f"frame{i:06}.png"
            save_path = os.path.join(bg_est_path, name)
            utils.save_image(save_path, bg)
            i = i +1 
    print("----------------------background estimation done----------------------")
    return bg_est_path

def main(args,**kwargs):
    ## Params and Settings 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    ## Data
    train_transforms = transforms.Compose([
        transforms.Resize(size=args.mask_shape),
        transforms.ToTensor()])
    root = os.path.join(args.parent_dir, args.dir, "frames")
    print(f"working on dataset in:{root}")
    predict_loader = DataLoader(
        ImageDataset(root=root,
                    transform=train_transforms),
        batch_size=1,#args.AE_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    # Model 
    checkpoint_path = os.path.join(args.BMN_ckpt_dir, args.BMN_ckpt)
    checkpoint = torch.load(checkpoint_path)
    # STN
    if "STN.homography_theta_regressor.0.weight" in checkpoint['state_dict'].keys():
        args.homography = True
        args.TG = "Homo"
        stn = STN_Homo.STN_Homo(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                    use_homography=args.homography).to(device) 
    else:                
        stn = STN.STN_block(args.mask_shape, args.pad, args.t, pretrained=args.pretrain_resnet,
                    use_homography=args.homography).to(device) 
    # AE  
    in_chan = args.channels*(args.moments_num + 1)
    ae = AutoEncoder(C=args.C,
                        M=args.M,
                        in_chan=in_chan,
                        out_chan=args.channels,
                        input_shape=args.mask_shape,
                        code_size=args.code_size).to(device)
    # BMN using STN and AE 
    bmn = BMN(args.moments_num, stn, ae, use_theta_embedding=args.theta_embedding,
              cond_decoder=args.cond_decoder).to(device)
    bmn.load_state_dict(checkpoint['state_dict'])
    print("BMN was loaded from: ", checkpoint_path)
    bmn.init_moments(predict_loader,args.trim_percentage)

    dataset_results_dir = os.path.join(args.Results_dir,args.dir)
    exp = args.log_name
    utils.safe_mkdir(dataset_results_dir)
    exp_path = os.path.join(dataset_results_dir, exp)
    bg_est_path = predict_BMN(bmn,predict_loader,exp_path)
    gt_path = os.path.join(args.parent_dir,args.dir,"GT")    
    video_path = os.path.join(args.parent_dir, args.dir, "frames")
    mse_path = os.path.join(exp_path, "MSE")
    
    calc_metric_and_MSE(video_path=video_path, bg_path=bg_est_path,
                        gt_path=gt_path, mse_path=mse_path, args=args, method=args.method,
                        overwrite=True)

if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    main(args)