from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from fastmri import rss
import h5py
import os
import numpy as np
import torch
from torch import fft as tfft
from torch.nn.functional import interpolate
from functools import lru_cache

__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, train: bool
):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )
    return dataloader


def normalize_complex(data, eps=1e-8):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std


def ifft(x: torch.Tensor) -> torch.Tensor:
    x = tfft.ifftshift(x, dim=[-2, -1])
    x = tfft.ifft2(x, dim=[-2, -1], norm="ortho")
    x = tfft.fftshift(x, dim=[-2, -1])
    return x


def make_gaussian_kernel(ksize: int, sigma: float = 0.5) -> torch.Tensor:
    x = torch.linspace(-ksize // 2 + 1, ksize // 2, ksize)
    x = x.expand(ksize, -1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


def estimate_sensitivity_maps_smooth(image_complex, eps=1e-6):
    """
    Estimate sensitivity maps using adaptive combine method.
    """
    if len(image_complex.shape) != 4:
        image_complex = image_complex.unsqueeze(0)
    # Compute RSS
    rss_image = rss(image_complex, dim=1)

    # Estimate initial sensitivities
    sens_maps = image_complex / (rss_image.unsqueeze(1) + eps)
    f, coil, h, w = sens_maps.shape
    # print(shape)

    # Apply Gaussian smoothing (this is a simplified version, consider using proper 2D Gaussian filter)
    kernel_size = 5
    kernel = make_gaussian_kernel(kernel_size, sigma=0.5)[None, None, ...]
    # print(kernel.shape)
    sens_maps = sens_maps.view(f * coil, 1, h, w)

    real_smooth = torch.nn.functional.conv2d(
        sens_maps.real, kernel, padding=kernel_size // 2
    )
    imag_smooth = torch.nn.functional.conv2d(
        sens_maps.imag, kernel, padding=kernel_size // 2
    )

    sens_maps_smooth = torch.complex(real_smooth, imag_smooth)
    sens_maps_smooth = sens_maps_smooth.view(f, coil, h, w)

    # Normalize smoothed sensitivity maps
    sens_maps_norm = sens_maps_smooth / (
        torch.sum(torch.abs(sens_maps_smooth) ** 2, dim=1, keepdim=True).sqrt() + eps
    )

    return sens_maps_norm.squeeze(0)


def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, "r") as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data


def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

@lru_cache(maxsize=20)
def load_kdata(filename):
    """
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    """
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata["real"] + 1j * kdata["imag"]
    return kdata


@register_dataset(name="recon")
class ReconDataset(Dataset):
    def __init__(
        self,
        root: str,
        mask_path: str,
        single_file_eval: bool = False,
    ) -> None:
        super().__init__()
        # print(f"Loading Dataset from: {root} and masks from {mask_path}")
        self.files = root
        self.sfe = single_file_eval
        self.mask = mask_path
        self.build()

    def build(self):
        path, name = os.path.split(self.files)
        # path = path to dataset
        # name = P###_T1/2map_slice_contrast.npy
        pid, cont, slice_no, ext = name.split('_')
        slice_no = int(slice_no)

        # check slice + 1, + 2 exist
        if os.path.exists(os.path.join(path, '_'.join([pid, cont, str(slice_no+1), ext]))):
            print('s+1 exists')
            if os.path.exists(os.path.join(path, '_'.join([pid, cont, str(slice_no+2), ext]))):
                print('s+2 exists')
                f2 = slice_no + 1
                f3 = slice_no + 2
            else:
                print('s+2 !exists')
                # s+1 exists, but not s+2
                f2 = slice_no + 1
                f3 = slice_no - 1
        else:
            f2 = slice_no - 1
            f3 = slice_no - 2
        
        self.path = path
        self.fname = name
        self.frames = [slice_no, f2, f3]
        self.cont = cont
        self.ext = ext
        self.f_id = pid

    def __len__(self):
        return 1

    def __getitem__(self, index):

        _1, _2, _3 = self.frames
        f1_path = os.path.join(self.path, '_'.join([self.f_id, self.cont, str(_1), self.ext]))
        f2_path = os.path.join(self.path, '_'.join([self.f_id, self.cont, str(_2), self.ext]))
        f3_path = os.path.join(self.path, '_'.join([self.f_id, self.cont, str(_3), self.ext]))

        # multi-coil images 10 x h x w
        frame0 = torch.from_numpy(np.load(f1_path, allow_pickle=True)[()]['complex-image-space'])
        csm_frame0 = estimate_sensitivity_maps_smooth(frame0.unsqueeze(0)).squeeze()
        frame0 = torch.sum(frame0 * csm_frame0.conj(), dim=0).numpy()
        frame1 = torch.from_numpy(np.load(f2_path, allow_pickle=True)[()]['complex-image-space'])
        csm_frame1 = estimate_sensitivity_maps_smooth(frame1.unsqueeze(0)).squeeze()
        frame1 = torch.sum(frame1 * csm_frame1.conj(), dim=0).numpy()
        frame2 = torch.from_numpy(np.load(f3_path, allow_pickle=True)[()]['complex-image-space'])
        csm_frame2 = estimate_sensitivity_maps_smooth(frame2.unsqueeze(0)).squeeze()
        frame2 = torch.sum(frame2 * csm_frame2.conj(), dim=0).numpy()

        # multi-coil images 10 x h x w
        # frame0 = torch.from_numpy(np.load(f1_path, allow_pickle=True)[()]['complex-image-space'])
        # csm_frame0 = torch.from_numpy(np.load(f1_path, allow_pickle=True)[()]['csm-4x'])
        # frame0 = torch.sum(frame0 * csm_frame0.conj(), dim=0).numpy()
        # frame1 = torch.from_numpy(np.load(f2_path, allow_pickle=True)[()]['complex-image-space'])
        # csm_frame1 = torch.from_numpy(np.load(f2_path, allow_pickle=True)[()]['csm-4x'])
        # frame1 = torch.sum(frame1 * csm_frame1.conj(), dim=0).numpy()
        # frame2 = torch.from_numpy(np.load(f3_path, allow_pickle=True)[()]['complex-image-space'])
        # csm_frame2 = torch.from_numpy(np.load(f3_path, allow_pickle=True)[()]['csm-4x'])
        # frame2 = torch.sum(frame2 * csm_frame2.conj(), dim=0).numpy()


        frame0, _ = normalize_complex(frame0)
        frame1, _ = normalize_complex(frame1)
        frame2, _ = normalize_complex(frame2)

        real0 = np.real(frame0)
        imag0 = np.imag(frame0)

        real1 = np.real(frame1)
        imag1 = np.imag(frame1)

        real2 = np.real(frame2)
        imag2 = np.imag(frame2)

        masks = torch.from_numpy(self.mask)
        masks = torch.stack([masks, masks, masks])

        out = np.stack([real0, imag0, real1, imag1, real2, imag2]).astype(np.float32)
        out = torch.from_numpy(out)
        max_val = abs(out).max()
        out /= max_val
        # out = resize(out, (320, 320), antialias=True, interpolation=0)
        h, w = out.shape[-2:]
        gt_image = out.clone()
        out = interpolate(out.unsqueeze(0), (160, 512), mode="nearest-exact").squeeze()

        csm_output = torch.from_numpy(np.stack([csm_frame0, csm_frame1, csm_frame2]))

        guidance = {'pid': self.f_id, 'name': self.fname, 'sense_maps': csm_output, 'mask': masks, 'shape': [h, w], 'gt_image': gt_image}

        return out.float(), guidance



@register_dataset(name="ffhq")
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + "/**/*.png", recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")

        return img


if __name__ == "__main__":
    from guided_diffusion.measurements import ReconOperatorSingle

    root = "/bigdata/CMRxRecon2023/test/multi_coil/P007_T2map_4_1.npy"
    mask_path = np.load('/home/anurag/Code/DiffuseRecon/acc4-t2-mask.npy')
    # mask_type = "ktGaussian24"
    # mask_type = "ktUniform24"
    sfe = True

    dataset = ReconDataset(
        root=root, mask_path=mask_path, single_file_eval=sfe
    )
    opp = ReconOperatorSingle()

    print("Len:", len(dataset))

    a, guide = dataset[12]

    print("\nBatch ->")
    print(type(a))
    print(a.shape, a.min(), a.max(), a.mean(), a.std())

    print(guide['pid'], guide['name'], dataset.frames)

    # Coil Sense Maps
    d = guide['sense_maps']
    print("\nCoil Sense Maps [est]")
    print(type(d))
    print(d.shape)

    # Masks
    f = guide['mask']
    print("\nMasks")
    print(type(f))
    print(f.shape)

    y = opp.A(a.unsqueeze(0), f.unsqueeze(0), d)

    z = opp.At(y, d, a.shape[-2: ])

    gt = guide['gt_image']

    np.save("input.npy", a.numpy())
    np.save("kspace.npy", gt)
    np.save("mask.npy", f)
    np.save("csm.npy", d)
    np.save("output.npy", y.numpy())
    np.save("inverse.npy", z.numpy())

    dataloader = get_dataloader(dataset, batch_size=2, train=False, num_workers=1)

    for a, b in dataloader:
        print(a.shape)
        print(len(b))

