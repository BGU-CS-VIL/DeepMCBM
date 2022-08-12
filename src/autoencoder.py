import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from gdn import GDN

# https://arxiv.org/pdf/1611.01704.pdf 
# A simplfied version without quantization
class AutoEncoder(nn.Module):
    def __init__(self, C=128, M=128, in_chan=3, out_chan=3,input_shape = (128,256),code_size=96):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan,input_shape=input_shape,code_size=code_size)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x, **kargs):
        code = self.encoder(x)
        out = self.decoder(code)
        return out
    
    def predict(self,image):
        with torch.no_grad():
            background = self.forward(image)
            return background

class Encoder(nn.Module):
    """ Encoder
    """
    def __init__(self, C=32, M=128, in_chan=3,input_shape = (128,256),code_size=96):
        super(Encoder, self).__init__()
        out_size = [C,int(input_shape[0]/32),int(input_shape[1]/32)]
        self.enc = nn.Sequential(
            # /2
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            # /4
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            # /8
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            # /16
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            # /32 
            nn.Conv2d(in_channels=M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False)
        )
        self.fc = nn.Sequential(nn.Linear(math.prod(out_size),code_size),
                                nn.LeakyReLU(True),
                                nn.Linear(code_size,math.prod(out_size)))
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=out_size)

    def forward(self, x):
        x = self.enc(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.unflatten(x)
        return x

class Decoder(nn.Module):
    """ Decoder
    """
    def __init__(self, C=32, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
        )
    
    def forward(self, q):
        return self.dec(q)












        



    
