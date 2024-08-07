"""This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n."""

from abc import ABC, abstractmethod
from typing import List
from torch.nn import functional as F
from torchvision import torch

from torch import fft as tfft

def fft(x: torch.Tensor) -> torch.Tensor:
    x = tfft.ifftshift(x)
    x = tfft.fft2(x, norm='ortho')
    x = tfft.fftshift(x)
    return x

def ifft(x: torch.Tensor) -> torch.Tensor:
    x = tfft.fftshift(x)
    x = tfft.ifft2(x, norm='ortho')
    x = tfft.ifftshift(x)
    return x
# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs) -> torch.Tensor:
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs) -> torch.Tensor:
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


class ReconOperatorSingle(LinearOperator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, **kwargs):
        pass

    def transpose(self, data, **kwargs):
        pass

    def A(
        self, image_space: torch.Tensor, mask: torch.Tensor, csms: torch.Tensor
    ) -> torch.Tensor:
        """
        image_spage: [b, 6, bh, hw] -> Image space data
                     6 = [r0, i0, r1, i1, r2, i2]
        mask       : [3, th, tw] -> 3 Frames, target H, target W
        csm        : [3, 10, ch, cw] -> One sense map per frame, 10 senses total
        """
        nf, th, tw = mask.shape
        image_space = F.interpolate(image_space, size=(th, tw), mode="nearest-exact")
        # [b, 6, th, tw]
        f0 = image_space[:, 0:2]
        f1 = image_space[:, 2:4]
        f2 = image_space[:, 4:6]
        image_space = (
            torch.stack([f0, f1, f2], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        )
        image_space = torch.view_as_complex(image_space)
        assert image_space.shape[1] == nf, "Cannot recover all frames correctly"
        image_senes = csms.unsqueeze(0) * image_space.unsqueeze(2)
        kspace_image = fft(image_senes)
        mask = mask.unsqueeze(1).repeat(1,10,1,1)
        kspace_image = kspace_image * mask
        return kspace_image

    def At(self, kspace_image: torch.Tensor, csm: torch.Tensor, dest_size: List[int]) -> torch.Tensor:
        """
        kspace_image: [b, frames, coils, th, tw] -> K-space data
        csms: [3, 10, ch, cw] -> One sense map per frame, 10 total
        dest_size: [int, int] -> resize image target
        """
        
        image_space = ifft(kspace_image)
        image_space = torch.sum(image_space * csm.unsqueeze(0).conj(), dim=2)
        f0 = image_space[:, 0]
        f1 = image_space[:, 1]
        f2 = image_space[:, 2]
        image_space = torch.stack([
            f0.real, f0.imag,
            f1.real, f1.imag,
            f2.real, f2.imag
        ],
        dim=1
        )
        image_space = F.interpolate(image_space, tuple(dest_size), mode='nearest-exact')
        return image_space


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):  # type: ignore
        """
        Follow skimage.util.random_noise.
        """

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np

        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(
            np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate
        )
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)
