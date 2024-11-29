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