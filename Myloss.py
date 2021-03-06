import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
#import pytorch_colors as colors
import numpy as np

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        #bias = 0.00001
        #x = torch.div(x_o,x_i+bias)
        #x = x_o
        b,c,h,w = x.shape
        '''
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        '''
        r, g, b = torch.split(x, 1, dim=1)
        pow_r = torch.norm(r, dim=(2,3))
        pow_g = torch.norm(g, dim=(2,3))
        pow_b = torch.norm(b, dim=(2,3))

        return (torch.pow(pow_r - pow_g, 2) + torch.pow(pow_r - pow_b, 2) + torch.pow(pow_b - pow_g, 2)) / (h*w)


def weights_map(Size):
    V=np.zeros(Size)
    x = np.append(np.matrix(np.arange(0,8)).getA(),np.matrix(np.arange(7,-1,-1)).getA())
    for i in range(0,Size[0]):
        for j in range(0,Size[1]):
            V[i][j] = math.log(math.e + math.sqrt(x[i]**2+x[j]**2))
    return torch.from_numpy(V).cuda()
class L_cen(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_cen, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow((mean- torch.FloatTensor([self.mean_val] ).cuda())* weights_map((16,16)).squeeze(0).squeeze(0) ,2))
        return d



class L_ill(nn.Module):
    def __init__(self):
        super(L_ill,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        #x = x.cpu()
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


class noise_loss(nn.Module):
    def __init__(self):
        super(noise_loss, self).__init__()
    
    def forward(self, x):
        batch_size = x.size()[0]
        loss = torch.pow(torch.norm(x, dim=(2,3)), 2)
        return torch.mean(loss,1)