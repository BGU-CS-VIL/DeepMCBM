import torch
import torch.nn as nn
import utils 
import matplotlib.pyplot as plt
class Alignment_Loss(nn.Module):
    def __init__(self,input_shape,memory=0.0, SmoothL1Loss_beta=0.35,zero_sensitivity=1e-10):
        super(Alignment_Loss, self).__init__()
        self.memory = memory 
        self.zero_sensitivity = zero_sensitivity
        self.huber = torch.nn.SmoothL1Loss(beta=SmoothL1Loss_beta,reduction='none')

        self.image_acc_sum = nn.Parameter(torch.zeros((3,input_shape[-2],input_shape[-1])),requires_grad=False)
        self.mask_acc_sum = nn.Parameter(torch.zeros((3,input_shape[-2],input_shape[-1])),requires_grad=False)
        self.mu = nn.Parameter(torch.zeros_like(self.image_acc_sum,device=self.image_acc_sum.device),requires_grad=False)
        self.mu_batch = nn.Parameter(torch.zeros_like(self.image_acc_sum,device=self.image_acc_sum.device),requires_grad=False)

    def forward(self,image,mask):
        with torch.no_grad():
            self.image_acc_sum+=image.sum(dim=0)
            self.mask_acc_sum+=mask.sum(dim=0)
            utils.weighted_average(self.mu,
                                self.image_acc_sum,
                                self.mask_acc_sum,
                                self.zero_sensitivity)
         
        diff_from_mu = self.huber(image,self.mu)*mask   # batch_size x C x H x W 
        loss = diff_from_mu.sum(dim = (1,2,3))          # batch size x 1
        mask_sum = mask.sum(dim = (1,2,3))              # batch size x 1
        loss = loss.div(mask_sum)                       # batch size x 1
        loss = loss.mean()                              # scalar
        return loss

    def step(self):
        self.image_acc_sum.mul_(self.memory)
        self.mask_acc_sum.mul_(self.memory)

    def init_loss(self,data_loader,model):
        with torch.no_grad():
            for image in data_loader:
                image = image.cuda()
                image_out,mask_out,theta_out = model(image)
                self.image_acc_sum.add_(image_out.sum(dim=0))
                self.mask_acc_sum.add_(mask_out.sum(dim=0))            
                torch.cuda.empty_cache()
            utils.weighted_average(self.mu,
                        self.image_acc_sum,
                        self.mask_acc_sum,
                        self.zero_sensitivity)
                    