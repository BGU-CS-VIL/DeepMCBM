import warnings
from torchvision.models import resnet18
import torch
from torch import nn
import torch.nn.functional as F
from transforms import *
import math
from utils import freeze_layers, unfreeze_layers
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


class STN_block(nn.Module):
    def __init__(self, mask_shape, pad, t, use_homography=False,
                 pretrained=False, device="cuda"):
        super(STN_block, self).__init__()

        # size of original image without the zero padding
        self.mask_shape = mask_shape[-2:]
        self.pad = pad
        self.device = device
        # theta reggesor sizes
        affine_theta_dim = 6
        self.flatten = nn.Flatten()
        self.backone = resnet18(pretrained=pretrained)
        self.out_shape = self.Backbone_output_size()
        self.linear_in_features = math.prod(self.out_shape)
        self.hidden_size = 32

        self.affine_theta_regressor = nn.Sequential(
            nn.Linear(self.linear_in_features, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, affine_theta_dim),
        )
        # other transforms
        self.use_homography = use_homography

        # homography
        if self.use_homography:
            self.homography_theta_dim = 8
            self.homography_theta_regressor = nn.Sequential(
                nn.Linear(self.linear_in_features, self.hidden_size),
                nn.ReLU(True),
                nn.Linear(self.hidden_size, self.homography_theta_dim),
            )

        # initialize reggresor weights and bias to zero to start from the identity tramsform
        with torch.no_grad():
            # filling with "zero" to resemble the identity transform as initializtion
            nn.init.normal_(self.affine_theta_regressor[-1].weight, std=1e-5)
            nn.init.normal_(self.affine_theta_regressor[-1].bias, std=1e-5)
            if self.use_homography:
                # TODO need different initialization for exp(homography) in Lie Algebra
                nn.init.normal_(
                    self.homography_theta_regressor[-1].weight, std=1e-5)
                # init with identity
                self.homography_theta_regressor[-1].bias.data.copy_(
                    torch.eye(3).flatten()[:self.homography_theta_regressor[-1].out_features])
        # global centering transformation
        self.global_transform = nn.Parameter(torch.tensor(
                                            [[1, 0, t[0]], 
                                             [0, 1, t[1]], 
                                             [0, 0, 1   ]]), requires_grad=False)
        # mean theta

    def forward(self, image):
        # get theta - always using affine regressor
        x = self.backone(image)
        x = self.flatten(x)
        image,mask = self.pad_image_and_mask(image)

        # TRANSFORMS
        transform = {}
        shape = image.size()
        N, C, H, W = shape

        # affine + global transform
        theta = self.affine_theta_regressor(x)
        transform["affine"] = theta
        _, grid = affine_warp(
            theta, shape, exp=True, grid=None, global_transform=self.global_transform, device=self.device)

        if self.use_homography:
            theta_homo = self.homography_theta_regressor(x)
            transform["homography"] = theta_homo
            _, grid = homography_warp(
                theta_homo, shape, exp=False, grid=grid, evice=self.device)

        # interpolation
        grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)  # for F.grid_sample
        warped_image = F.grid_sample(image, grid)
        warped_mask = F.grid_sample(mask, grid)
        
        return warped_image, warped_mask, transform

    def pad_image_and_mask(self,image):
        # padding
        image = F.pad(image,
                      (0,          # left
                       self.pad[1],  # right
                       0,           # top
                       self.pad[0]  # bottom
                       ))
        # create mask
        mask = F.pad(torch.ones(self.mask_shape, device=image.device),
                     (
            0,           # left
            self.pad[1],  # right
            0,           # top
            self.pad[0]  # bottom
        )).expand_as(image)

        return image,mask

    def Backbone_output_size(self):
        x = torch.rand((1, 3, *self.mask_shape))
        x = self.backone(x)
        return x.shape

    def update_global_transform(self,delta):
        self.global_transform.data += delta

    def freeze(self, layer_names):
        freeze_layers(self, layer_names)

    def unfreeze(self, layer_names):
        unfreeze_layers(self, layer_names)
    
    def set_use_homography(self, use=True):
        self.use_homography = use
    
    def forward_1(self, image):
        # use another function in BMN to enable torch summary
        return self.forward(image)