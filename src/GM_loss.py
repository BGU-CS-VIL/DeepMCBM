import torch 
import torch.nn as nn 

class GM(nn.Module):
    def __init__(self,sigma,reduction='mean'):
        super(GM, self).__init__()
        self.sigma = sigma 
        self.reduction = reduction

    def forward(self,x,y):
        obj = torch.square(x-y)/(torch.square(x-y)+self.sigma**2)
        if self.reduction == 'mean':
            return torch.mean(obj)
        elif self.reduction =='sum':
            return torch.sum(obj)
        elif self.reduction == 'None':
            return obj 