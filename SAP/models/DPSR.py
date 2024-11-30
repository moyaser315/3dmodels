import torch
from torch import nn
from ..utils import utils

class DPSR(nn.Module):
    def __init__(self, res_grid, sigma):
        super(DPSR,self).__init__()
        self.res_grid = res_grid
        self.sigma = sigma
        self.m = len(res_grid)
        self.freqs = utils.get_fft(res_grid)
        self.G = utils.get_smoothed_guassian(self.freqs,self.res,sigma)

    def forward(self, points , normals):

        v = None #TODO imp reasterization 
        batch,dim,x,y,z = 0,1,2,3,4
        # divergence (first drevitive) : 

        #prepering vector V to be computed in frequency domaibn
        dv = torch.fft.rfft(v,dim=(x,y,z))
        dv = dv.permute([batch,x,y,z,dim])
        dv = dv.unsqueeze(-1) * self.G

        freqs = self.freqs.unsqueeze(-1)
        freqs *= (2*torch.pi)
        freqs = freqs.to(points.device)

        #computing dv
        dv = torch.sum(torch.view_as_real(dv.unsqueeze(-1)) * freqs , dim=-2) #TODO:handle complex numbers

        #Laplacian (second derivative) :
        lap = -torch.sum()




        
        

