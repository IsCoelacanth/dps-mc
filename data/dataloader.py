from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from fastmri import rss_complex
import h5py
import os
import numpy as np
import torch
from torch import fft as tfft
from torch.nn.functional import interpolate
from functools import lru_cache
import torch.nn.functional as F
import torch.fft as fft
from .sens_maps import espirit
from .espirit import estimate_sensitivity_map
from tqdm.auto import tqdm

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


def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int, train: bool):
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


def estimate_sensitivity_maps_smooth(image_complex, rsimage=None, eps=1e-6):
    """
    Estimate sensitivity maps using adaptive combine method.
    """
    if rsimage is not None:
        if rsimage.shape[1] == 10:
            rss_image = rss_complex(torch.view_as_real(rsimage.contiguous()), dim=1)
        else:
            rss_image = torch.abs(torch.sum(rsimage, dim=1))
    else:
        rss_image = rss_complex(torch.view_as_real(image_complex.contiguous()), dim=1)

    # Estimate initial sensitivities
    sens_maps = image_complex / (rss_image.unsqueeze(1) + eps)
    f, coil, h, w = sens_maps.shape

    kernel_size = 5
    kernel = make_gaussian_kernel(kernel_size, sigma=1)[None, None, ...]
    kernel = kernel.to(image_complex.device)
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

    return sens_maps_norm


def smoothen_sense_maps(sens_maps):
    f, coil, h, w = sens_maps.shape
    kernel_size = 7
    kernel = make_gaussian_kernel(kernel_size, sigma=1)[None, None, ...]
    kernel = kernel.to(sens_maps.device)
    sens_maps = sens_maps.view(f * coil, 1, h, w)

    real_smooth = torch.nn.functional.conv2d(
        sens_maps.real, kernel, padding=kernel_size // 2
    )
    imag_smooth = torch.nn.functional.conv2d(
        sens_maps.imag, kernel, padding=kernel_size // 2
    )

    sens_maps_smooth = torch.complex(real_smooth, imag_smooth)
    sens_maps_smooth = sens_maps_smooth.view(f, coil, h, w)

    return sens_maps_smooth


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
        if not single_file_eval:
            self.files = glob(root + "/**/*.mat", recursive=True)
            self.files = sorted(self.files, key=lambda x: int(x.split("/")[-2][1:]))
        else:
            self.files = root
        self.sfe = single_file_eval
        self.mask_path = mask_path
        self.build()

    def build(self):
        if self.sfe:
            # single file eval mode
            # [frames, slices, coils, h, w]
            dataset = []
            kspace_data = load_kdata(self.files)

            # [frames, h, w]
            mask_data = loadmat(self.mask_path)["mask"]

            self.kdata = kspace_data
            self.mask = mask_data
            self.total_frames = kspace_data.shape[0]
            ns = kspace_data.shape[1]
            ns1 = ns // 2 - 1
            ns2 = ns // 2
            slice_list = [ns1, ns2]
            if ns == 1:
                ns1 = 0
                ns2 = 0
                slice_list = [0]

            jsize = kspace_data.shape[0] if "mapping" in self.files.lower() else 3
            for i in slice_list:
                # for j in range(jsize):
                for j in range(jsize):
                    dataset.append([j, i])
            self.dataset = dataset
        else:
            raise NotImplementedError("Only supports single item inference")

    def __len__(self):
        return len(self.dataset)

    def get_sense(self, kspace):
        sense_maps = torch.stack(
            [
                torch.from_numpy(
                    espirit(np.transpose(X, (1, 2, 0))[None, ...], 6, 16, 0.01, 0.5)
                ).permute(2, 0, 1)
                for X in kspace
            ]
        )
        return sense_maps

    def __getitem__(self, index):
        f0, sl = self.dataset[index]

        f1 = (f0 + 1) % self.total_frames
        f2 = (f0 + 2) % self.total_frames

        masks = torch.from_numpy(
            np.stack([self.mask[f0], self.mask[f1], self.mask[f2]])
        )

        current_kspace = self.kdata[:, sl][f0 : f2 + 1]
        # print(current_kspace.shape)
        # [12, C, H, W]
        image_space = torch.from_numpy(current_kspace)
        image_space = ifft(image_space)
        if "radial" in self.files.lower():
            # print("using radial")
            sense_maps = self.get_sense(current_kspace)
        else:
            # print("using cartesian")
            sense_maps = estimate_sensitivity_map(
                current_kspace.T, masks[0].numpy().T
            ).T
        sense_maps = smoothen_sense_maps(sense_maps)
        # sense_maps = sense_maps / torch.linalg.norm(sense_maps)
        sense_maps = sense_maps / (
                torch.sum(torch.abs(sense_maps) ** 2, dim=1, keepdim=True).sqrt() + 1e-8
            )

        fused = torch.mean(image_space * sense_maps.conj(), dim=1) #/ torch.linalg.norm(sense_maps.conj())
        fused = fused.numpy()

        frame0 = fused[0]
        frame1 = fused[1]
        frame2 = fused[2]

        frame0, mag0 = normalize_complex(frame0)
        frame1, mag1 = normalize_complex(frame1)
        frame2, mag2 = normalize_complex(frame2)

        real0 = np.real(frame0)
        imag0 = np.imag(frame0)

        real1 = np.real(frame1)
        imag1 = np.imag(frame1)

        real2 = np.real(frame2)
        imag2 = np.imag(frame2)

        out = np.stack([real0, imag0, real1, imag1, real2, imag2]).astype(np.float32)
        out = torch.from_numpy(out)
        max_val = abs(out).max()
        out /= max_val
        # out = resize(out, (320, 320), antialias=True, interpolation=0)
        h, w = out.shape[-2:]
        gt_image = out.clone()
        # np.save('outt.npy', out.cpu().numpy())
        out = interpolate(out.unsqueeze(0), (256, 512), mode="nearest-exact").squeeze()

        kspace_output = torch.from_numpy(
            np.stack(
                [
                    current_kspace[0],
                    current_kspace[1],
                    current_kspace[2],
                ]
            )
        )
        csm_output = torch.from_numpy(
            np.stack([sense_maps[0], sense_maps[1], sense_maps[2]])
        )

        guidance = {
            "slice_no": sl,
            "frame_no": f0,
            "sense_maps": csm_output,
            "kspace": kspace_output,
            "mask": masks,
            "shape": [h, w],
            "gt_image": gt_image,
            "stds": {
                "std1": torch.tensor(mag0),
                "std2": torch.tensor(mag1),
                "std3": torch.tensor(mag2),
            },
            "max": max_val.item(),
        }

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

    root = "/bigdata/CMRxRecon2024/ChallengeData/MultiCoil/Aorta/TrainingSet/FullSample/P002/aorta_sag.mat"
    mask_path = "/bigdata/CMRxRecon2024/ChallengeData/MultiCoil/Aorta/TrainingSet/Mask_Task2/P002/"
    mask_type = "ktRadial4"
    # mask_type = "ktGaussian24"
    # mask_type = "ktUniform24"
    sfe = True

    dataset = ReconDataset(
        root=root, mask_path=mask_path, us_mask_type=mask_type, single_file_eval=sfe
    )
    opp = ReconOperatorSingle()

    print("Len:", len(dataset))

    a, guide = dataset[12]

    print("\nBatch ->")
    print(type(a))
    print(a.shape, a.min(), a.max(), a.mean(), a.std())

    print(guide["slice_no"], guide["frame_no"])

    # Coil Sense Maps
    d = guide["sense_maps"]
    print("\nCoil Sense Maps [est]")
    print(type(d))
    print(d.shape)

    # Kspace
    e = guide["kspace"]
    print("\nKspace")
    print(type(e))
    print(e.shape)

    # Masks
    f = guide["mask"]
    print("\nMasks")
    print(type(f))
    print(f.shape)

    y = opp.A(a.unsqueeze(0), f, d)

    z = opp.At(y, d, a.shape[-2:])

    np.save("input.npy", a.numpy())
    np.save("kspace.npy", e)
    np.save("mask.npy", f)
    np.save("csm.npy", d)
    np.save("output.npy", y.numpy())
    np.save("inverse.npy", z.numpy())

    dataloader = get_dataloader(dataset, batch_size=2, train=False, num_workers=1)

    for a, b in dataloader:
        print(a.shape)
        print(len(b), b["slice_no"], b["frame_no"], b["kspace"].shape)
