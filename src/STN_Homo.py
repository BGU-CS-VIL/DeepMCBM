import warnings
from torchvision.models import resnet18
import torch
from torch import nn
import torch.nn.functional as F
from transforms import *
import math
from utils import freeze_layers, unfreeze_layers
import STN 
import os 
import utils 


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STN_Homo(STN.STN_block):
    def __init__(self, mask_shape, pad, t, use_homography=False,pretrained=False, device="cuda"):

        super().__init__(mask_shape, pad, t, use_homography,pretrained, device)

        self.homography_theta_dim = 8
        self.homography_theta_regressor = nn.Sequential(
            nn.Linear(self.linear_in_features, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.homography_theta_dim),
        )
    
        # TODO need different initialization for exp(homography) in Lie Algebra
        nn.init.normal_(
            self.homography_theta_regressor[-1].weight, std=1e-5)
        # init with identity
        self.homography_theta_regressor[-1].bias.data.copy_(
            torch.eye(3).flatten()[:self.homography_theta_regressor[-1].out_features])


    def forward(self, image):
    
        # TRANSFORMS
        x = self.backone(image)
        x = self.flatten(x)
        image,mask = self.pad_image_and_mask(image)

        transform = {}
        shape = image.size()
        N, C, H, W = shape

        # affine / diffeoAffine + global transform
        theta = self.affine_theta_regressor(x)
        transform["affine"] = theta
        _, grid = affine_warp(theta, shape, exp=True, grid=None, global_transform=self.global_transform, device=self.device)

        theta_homo = self.homography_theta_regressor(x)
        transform["homography"] = theta_homo
        _, grid = homography_warp(theta_homo, shape, exp=False, grid=grid,  device=self.device)

        # interpolation
        grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)  # for F.grid_sample
        warped_image = F.grid_sample(image, grid)
        warped_mask = F.grid_sample(mask, grid)
        
        return warped_image, warped_mask, transform

    def Load_Backbone_and_AffineHead(self,args):
            stn_a = STN.STN_block(args.mask_shape, args.pad, args.t, 
                                  pretrained=args.pretrain_resnet,
                                  use_homography=args.homography).to(device)
            checkpoint_path = os.path.join(args.STN_ckpt_dir, args.STN_ckpt)
            checkpoint = torch.load(checkpoint_path)
            stn_a.load_state_dict(checkpoint['state_dict'])
            self.backone = stn_a.backone
            print("backbone was loaded from:",checkpoint_path)
            self.affine_theta_regressor = stn_a.affine_theta_regressor
            print("Affine Regressor was loaded from:",checkpoint_path)
            # utils.freeze_layers(self,['backone','affine_theta_regressor'])
            
    
