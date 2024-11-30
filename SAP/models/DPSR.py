import torch
from torch import nn
from ..utils import utils


class DPSR(nn.Module):
    def __init__(self, res_grid, sigma):
        super(DPSR, self).__init__()
        self.res_grid = res_grid
        self.sigma = sigma
        self.m = len(res_grid)
        self.freqs = utils.get_fft(res_grid)
        self.G = utils.get_smoothed_guassian(self.freqs, self.res, sigma)

    def forward(self, points, normals):

        v = None  # TODO imp reasterization
        batch, dim, x, y, z = 0, 1, 2, 3, 4
        # divergence (first drevitive) :

        # prepering vector V to be computed in frequency domaibn
        dv = torch.fft.rfft(v, dim=(x, y, z))
        dv = dv.permute([batch, x, y, z, dim])
        dv = dv.unsqueeze(-1) * self.G

        freqs = self.freqs.unsqueeze(-1)
        freqs *= 2 * torch.pi
        freqs = freqs.to(points.device)

        # computing dv
        dv = torch.sum(
            -utils.get_img_sign(torch.view_as_real(dv[..., 0])) * freqs, dim=-2
        )  # TODO:handle complex numbers

        # Laplacian (second derivative) :
        lap = -torch.sum(freqs**2, dim=-2)  # x^2 + y^2 + z^2
        lap = dv / (lap + 1e-6)
        # NOTE: in the original implementation there's a permuate that i dopn't know why it's there
        lap = torch.fft.irfft(
            torch.view_as_complex(lap), s=self.res_grid, dim=(1, 2, 3)
        )

        return lap
