import warnings
from torchvision.models import resnet18
import torch
from torch import nn
import torch.nn.functional as F
from transforms import *
import math
from libcpab import Cpab
from utils import freeze_layers, unfreeze_layers
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")


class STN_block(nn.Module):
    def __init__(self, mask_shape, pad, t, use_homography=False, use_cpab=False,
                 tess_size=[2, 2], pretrained=False, zero_boundary=False, device="cuda"):
        super(STN_block, self).__init__()

        # size of original image without the zero padding
        self.mask_shape = mask_shape[-2:]
        self.pad = pad
        self.device = device
        # theta reggesor sizes
        affine_theta_dim = 6
        self.flatten = nn.Flatten()
        self.backone = resnet18(pretrained=pretrained)
        self.out_shape = self.back_bone_output_size()
        self.linear_in_features = math.prod(self.out_shape)
        self.hidden_size = 32

        self.affine_theta_regressor = nn.Sequential(
            nn.Linear(self.linear_in_features, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, affine_theta_dim),
        )
        # other transforms
        self.use_homography = use_homography
        self.use_cpab = use_cpab
        # for cpab
        self.T = None

        # homography
        if self.use_homography:
            self.homography_theta_dim = 8
            self.homography_theta_regressor = nn.Sequential(
                nn.Linear(self.linear_in_features, self.hidden_size),
                nn.ReLU(True),
                nn.Linear(self.hidden_size, self.homography_theta_dim),
            )
        # cpab
        if self.use_cpab:
            self.T = Cpab(tess_size=tess_size, backend="pytorch",
                          device="gpu", zero_boundary=zero_boundary)
            self.cpab_theta_dim = self.T.get_theta_dim()
            print("CPAB: theta shape: ", self.cpab_theta_dim)

            self.cpab_theta_regressor = nn.Sequential(

                nn.Linear(self.linear_in_features, self.hidden_size),
                nn.ReLU(True),
                nn.Linear(self.hidden_size, self.cpab_theta_dim),
            )

        # initialize reggresor weights and bias to zero to start from the identity tramsform
        with torch.no_grad():
            # filling with "zero" to resemble the identity transform as initializtion
            nn.init.normal_(self.affine_theta_regressor[-1].weight, std=1e-5)
            nn.init.normal_(self.affine_theta_regressor[-1].bias, std=1e-5)
            if self.use_cpab:
                nn.init.normal_(self.cpab_theta_regressor[-1].weight, std=1e-5)
                nn.init.normal_(self.cpab_theta_regressor[-1].bias, std=1e-5)
            if self.use_homography:
                # TODO need different initialization for exp(homography) in Lie Algebra
                nn.init.normal_(
                    self.homography_theta_regressor[-1].weight, std=1e-5)
                # init with identity
                self.homography_theta_regressor[-1].bias.data.copy_(
                    torch.eye(3).flatten()[:self.homography_theta_regressor[-1].out_features])
        # global centering transformation
        self.global_transform = nn.Parameter(torch.tensor(
            [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]), requires_grad=False)
        # mean theta
        self.mean_theta = nn.Parameter(torch.zeros(affine_theta_dim), requires_grad=False)

    def forward(self, image):
        
  
        # get theta - always using affine regressor
        x = self.backone(image)
        x = self.flatten(x)
        image,mask = self.pad_image_and_mask(image)


        # TRANSFORMS
        transform = {}
        shape = image.size()
        N, C, H, W = shape
        # affine / diffeoAffine + global transform
        theta = self.affine_theta_regressor(x)-self.mean_theta
        transform["affine"] = theta
        _, grid = affine_warp(
            theta, shape, exp=True, grid=None, global_transform=self.global_transform, device=self.device)

        if self.use_homography:
            theta_homo = self.homography_theta_regressor(x)
            transform["homography"] = theta_homo
            _, grid = homography_warp(
                theta_homo, shape, exp=False, grid=grid,  device=self.device)

        if self.use_cpab:
            theta_cpab = self.cpab_theta_regressor(x)
            transform["cpab"] = theta_cpab
            _, grid = cpab_warp(
                self.T, theta_cpab, shape, grid=grid)

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

    def back_bone_output_size(self):
        x = torch.rand((1, 3, *self.mask_shape))
        x = self.backone(x)
        return x.shape

    def forward_1(self, image):
        # use another function in BMN to enable torch summary
        return self.forward(image)

    def update_global_transform(self,delta):
        self.global_transform.data += delta


    def freeze(self, layer_names):
        freeze_layers(self, layer_names)

    def unfreeze(self, layer_names):
        unfreeze_layers(self, layer_names)
    
    def set_use_cpab(self, use=True):
        self.use_cpab = use
    
    def set_use_homography(self, use=True):
        self.use_homography = use
    
    def global_transform_zero_mean(self,data_loader:DataLoader):
        with torch.no_grad():
            theta_sum = torch.zeros_like(self.mean_theta)
            for image in data_loader:
                image = image.cuda()
                image_out,mask_out,theta_dict = self.forward(image)
                theta = theta_dict["affine"]
                theta_sum.add_(theta.sum(dim=0))
                torch.cuda.empty_cache()
            self.theta_mean = nn.Parameter(theta_sum.div_(data_loader.dataset.__len__()),requires_grad=False)
            print("mean:\n",self.theta_mean)

