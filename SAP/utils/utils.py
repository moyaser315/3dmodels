import torch
from typing import Tuple


def get_fft(res_grid: tuple) -> torch.tensor:
    freqs = []
    for res in res_grid:
        freqs.append(torch.fft.fftfreq(res, 1 / res))
    freqs = torch.meshgrid(freqs)
    freqs = list(freqs)
    freqs = torch.stack(freqs, dim=-1)
    return freqs


def get_smoothed_guassian(freqs, res, sigma):

    ecludian_dis = torch.sqrt(torch.sum(freqs**2, dim=-1))
    apply_filter = torch.exp(-0.5 * ((sigma * 2 * ecludian_dis / res[0]) ** 2))
    apply_filter = apply_filter.unsqueeze(-1).unsqueeze(
        -1
    )  # for brodcasting from the original code
    apply_filter.requires_grad = False

    return apply_filter


def get_img_sign(x, deg=1):
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res
