from cv2 import transform
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BMN(nn.Module):
    def __init__(self, moments_num, STN, AE, use_theta_embedding=True, cond_decoder=True):

        super(BMN, self).__init__()
        self.STN = STN
        self.AE = AE
        self.moments_num = moments_num
        self.use_theta_embedding = use_theta_embedding
        self.total_channels = 3*(self.moments_num+1)
        self.cond_decoder = cond_decoder

        moments_shape = (self.moments_num,
                        3,
                        self.STN.mask_shape[0]+self.STN.pad[0],
                        self.STN.mask_shape[1]+self.STN.pad[1])
        self.moments = torch.zeros(moments_shape, device=device)

        # freez STN :
        for param in self.STN.parameters():
            param.requires_grad = False
        self.STN.eval()

        self.theta_embeddings_linear = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),

        )
        self.up_sample_list = nn.ModuleList([nn.Sequential(nn.Conv2d(10, 10, 3, 1, 1),
                                                           nn.ReLU(True),
                                                           nn.Conv2d(
                                                               10, 10, 3, 1, 1),
                                                           nn.ReLU(True),
                                                           nn.Upsample(scale_factor=2)) for i in range(3)])
        self.theta_embeddings_conv = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1, 1),
            *self.up_sample_list,
            nn.Conv2d(10, 1, 3, 1, 1),
        )
        self.post_AE = nn.Sequential(
            nn.Conv2d(self.total_channels, 50, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(50, 3, kernel_size=3, padding=1))

    def forward(self, x, **kargs):
        # use another function to enable torch summary
        with torch.no_grad():
            warped_image, mask_out, transform = self.STN.forward_1(x)  
            moments_list = []
            for i in range(self.moments_num):
                input_moment = self.moments[i]
                input_moment = input_moment.expand(warped_image.shape)

                warped_moment = utils.warp_inv(input_moment, 
                                               theta_dict=transform, 
                                               shape=warped_image.shape,
                                               device=device,
                                               global_transform=self.STN.global_transform)[..., :x.shape[-2], :x.shape[-1]]
                moments_list.append(warped_moment)

            x = torch.cat((x, *moments_list), dim=1)
            if self.use_theta_embedding:
                theta_embedding_lin = self.theta_embeddings_linear(
                    transform.view(-1, 6))
                theta_embedding = self.theta_embeddings_conv(
                    theta_embedding_lin.view(-1, 1, 16, 32))
                x = torch.cat((x, theta_embedding), dim=1)
        AE_output = self.AE(x)
        if self.cond_decoder:
            x_cat_moments = torch.cat((AE_output, *moments_list), dim=1)
            x = AE_output + self.post_AE(x_cat_moments)
        else:
            x = AE_output
        reconstruction = torch.sigmoid(x)

        return reconstruction, AE_output, warped_image, transform, moments_list[0] 

    def init_moments(self,data_loader,trim_percentage):
        print("initializing BMN moments")
        self.fill_moments(data_loader,trim_percentage)
        delta = self.check_moments_boundry()
        # update only three times to avoid infinit loop and hope for the best 
        for i in range(3):
            if not (delta==0).all():
                print("alignment touches bouandry of the paded area,\nfixing global centering transform")
                print(f"attempt number {i+1}")
                self.STN.update_global_transform(delta)
                self.moments.fill_(0.0)
                self.fill_moments(data_loader,trim_percentage)
                delta = self.check_moments_boundry()
        return self.moments

    def fill_moments(self,data_loader,trim_percentage):
        if len(data_loader.dataset) > 300:
            print("Fill moments without trim")
            self.fill_moments_no_trim(data_loader)
        else:
            try:
                print("Try to fill moments with trim") 
                self.fill_moments_trim(data_loader,trim_percentage)
            except:      
                print("Can not trim")  
                print("Fill moments without trim")
                self.fill_moments_no_trim(data_loader)


    def fill_moments_trim(self,data_loader,trim_percentage):
        with torch.no_grad():
            self.STN.eval()
            # create pixel stack
            image_stack, mask_stack = utils.create_pixel_stack(
                data_loader, self.STN)
            # trim pixel stack
            image_stack, mask_stack = utils.trim_pixel_stack(
                image_stack, mask_stack,trim_percentage)
                
            mask_sum = mask_stack.sum(axis=0)
            for i, m in enumerate(self.moments):
                image_sum = (image_stack-self.moments[0]*mask_stack)**(i+1)
                image_sum = image_sum.sum(axis=0)
                utils.weighted_average(
                    self.moments[i], image_sum, mask_sum, zero_sensitivity=1e-5)
    
    def fill_moments_no_trim(self,data_loader):
        ## consume less memory, does not need to hold the entire pixel stack at once 
        with torch.no_grad():
            image = next(iter(data_loader))
            sample_image, sample_mask, transform = self.STN(image.cuda())
            moment_sum = torch.zeros_like(sample_image[0])
            mask_sum = torch.zeros_like(sample_mask[0])
            for i, m in enumerate(self.moments):
                for image in data_loader:
                     image_out, mask_out, transform = self.STN(image.cuda())
                     moment = (image_out-self.moments[0]*mask_out)**(i+1)
                     moment_sum+=moment.sum(dim=0)
                     mask_sum+=mask_out.sum(dim=0)
                utils.weighted_average(
                    self.moments[i], moment_sum, mask_sum, zero_sensitivity=1e-5)


    def check_moments_boundry(self):
        mu = self.moments[0]
        delta = torch.zeros((3,3), device=device)
        # touching top
        if mu[:,0,:].max() > 1e-5 :
            delta[1,2] += -0.1
        # touching bottom
        if mu[:,-1,:].max() > 1e-5 :
            delta[1,2] += 0.1
        # touching left
        if mu[:,:,0].max() > 1e-5 :
            delta[0,2] += -0.1
        # touching right
        if mu[:,:,-1].max() > 1e-5 :
            delta[0,2] += 0.1 
        return delta

    def predict(self,image):
        with torch.no_grad():
            background, AE_output, warped_image,transform,warped_mean = self.forward(image)
            return background,warped_mean