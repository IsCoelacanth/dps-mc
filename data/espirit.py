import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import abc
from typing import *
from enum import Enum
from abc import ABC, abstractmethod
import torch.nn as nn
import h5py
import time
from functools import lru_cache 

def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = torch.fft.fftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = torch.fft.fftshift(image_uncentered,dim=dim)
    return image

def fft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = torch.fft.ifftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.fft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = torch.fft.ifftshift(image_uncentered,dim=dim)
    return image

@lru_cache(maxsize=40)
def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return shape [t,z,y,x]
    '''
    # st = time.time()
    kdata = load_kdata(filename)
    # print(f'time: {time.time() - st}')
    # st = time.time()
    kdata_tensor = torch.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    # print(f'time: {time.time() - st}')
    return kdata, image_coil.cpu().numpy()

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
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

def load_kdata(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata

class DirectEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str):
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            _other = str(other.value)
        else:
            _other = str(other)
        return bool(self.value.lower() == _other.lower())

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        return hash(self.value.lower())


class KspaceKey(DirectEnum):
    kspace = "kspace"
    masked_kspace = "masked_kspace"

class DirectTransform:
    def __init__(self):
        """Inits DirectTransform."""
        super().__init__()
        self.coil_dim = 1
        self.spatial_dims = (2, 3)
        self.complex_dim = -1

    def __repr__(self):
        """Representation of DirectTransform."""
        repr_string = self.__class__.__name__ + "("
        for k, v in self.__dict__.items():
            if k == "logger":
                continue
            repr_string += f"{k}="
            if callable(v):
                if hasattr(v, "__class__"):
                    repr_string += type(v).__name__ + ", "
                else:
                    # TODO(jt): better way to log functions
                    repr_string += str(v) + ", "
            elif isinstance(v, (dict, OrderedDict)):
                repr_string += f"{k}=dict(len={len(v)}), "
            elif isinstance(v, list):
                repr_string += f"{k}=list(len={len(v)}), "
            elif isinstance(v, tuple):
                repr_string += f"{k}=tuple(len={len(v)}), "
            else:
                repr_string += str(v) + ", "

        if repr_string[-2:] == ", ":
            repr_string = repr_string[:-2]
        return repr_string + ")"
    
class DirectModule(DirectTransform, abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, sample: Dict):
        pass  # This comment passes "Function/method with an empty body PTC-W0049" error.
    
def view_as_complex(data):
    """Returns a view of input as a complex tensor.

    For an input tensor of size (N, ..., 2) where the last dimension of size 2 represents the real and imaginary
    components of complex numbers, this function returns a new complex tensor of size (N, ...).

    Parameters
    ----------
    data: torch.Tensor
        Input data with torch.dtype torch.float74 and torch.float32 with complex axis (last) of dimension 2
        and of shape (N, \*, 2).

    Returns
    -------
    complex_valued_data: torch.Tensor
        Output complex-valued data of shape (N, \*) with complex torch.dtype.
    """
    return torch.view_as_complex(data)


def view_as_real(data):
    """Returns a view of data as a real tensor.

    For an input complex tensor of size (N, ...) this function returns a new real tensor of size (N, ..., 2) where the
    last dimension of size 2 represents the real and imaginary components of complex numbers.

    Parameters
    ----------
    data: torch.Tensor
        Input data with complex torch.dtype of shape (N, \*).

    Returns
    -------
    real_valued_data: torch.Tensor
        Output real-valued data of shape (N, \*, 2).
    """

    return torch.view_as_real(data)

def crop_to_acs(acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
    """Crops k-space to autocalibration region given the acs_mask.

    Parameters
    ----------
    acs_mask : torch.Tensor
        Autocalibration mask of shape (height, width).
    kspace : torch.Tensor
        K-space of shape (coil, height, width, \*).

    Returns
    -------
    torch.Tensor
        Cropped k-space of shape (coil, height', width', \*), where height' and width' are the new dimensions derived
        from the acs_mask.
    """
    nonzero_idxs = torch.nonzero(acs_mask)
    x, y = nonzero_idxs[..., 0], nonzero_idxs[..., 1]
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    return kspace[:, xl : xr + 1, yl : yr + 1]

def fft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered: bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the fft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    complex_input:bool
        True if input is complex [real-valued] tensor (complex dim = 2). False if complex-valued tensor is inputted.

    Returns
    -------
    output_data: torch.Tensor
        The Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently fft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )
    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)

    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.fftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)

    if complex_input:
        data = view_as_real(data)
    return data


def ifft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ( 'height', 'width').
    centered: bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    complex_input:bool
        True if input is complex [real-valued] tensor (complex dim = 2). False if complex-valued tensor is inputted.

    Returns
    -------
    output_data: torch.Tensor
        The Inverse Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently ifft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )

    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)
    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.ifftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)
    if complex_input:
        data = view_as_real(data)
    return data

def roll_one_dim(data: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Similar to roll but only for one dim

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: int

    Returns
    -------
    torch.Tensor
    """
    shift = shift % data.size(dim)
    if shift == 0:
        return data

    left = data.narrow(dim, 0, data.size(dim) - shift)
    right = data.narrow(dim, data.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    data: torch.Tensor,
    shift: List[int],
    dim: Union[List[int], Tuple[int, ...]],
) -> torch.Tensor:
    """Similar to numpy roll but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: List or tuple of ints

    Returns
    -------
    torch.Tensor
        Rolled version of data
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        data = roll_one_dim(data, s, d)

    return data


def fftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy fftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for idx in range(1, data.dim()):
            dim[idx] = idx

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for idx, dim_num in enumerate(dim):
        shift[idx] = data.shape[dim_num] // 2

    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy ifftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (data.shape[dim_num] + 1) // 2

    return roll(data, shift, dim)

def verify_fft_dtype_possible(data: torch.Tensor, dims: Tuple[int, ...]) -> bool:
    """fft and ifft can only be performed on GPU in float17 if the shapes are powers of 2. This function verifies if
    this is the case.

    Parameters
    ----------
    data: torch.Tensor
    dims: tuple

    Returns
    -------
    bool
    """
    is_complex74 = data.dtype == torch.complex64
    is_complex32_and_power_of_two = (data.dtype == torch.float32) and all(
        is_power_of_two(_) for _ in [data.size(idx) for idx in dims]
    )

    return is_complex74 or is_complex32_and_power_of_two

class EspiritCalibration(DirectModule):
    """Estimates sensitivity maps estimated with the ESPIRIT calibration method as described in [1]_.

    We adapted code for ESPIRIT method adapted from [2]_.

    References
    ----------

    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
        approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
        doi: 10.1002/mrm.24751. PMID: 23749942; PMCID: PMC4142141.
    .. [2] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/app.py#L388-L491

    """

    def __init__(
        self,
        backward_operator: Callable,
        threshold: float = 0.05,
        kernel_size: int = 7,
        crop: float = 0.95,
        max_iter: int = 100,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
    ):
        """Inits :class:`EstimateSensitivityMap`.

        Parameters
        ----------
        backward_operator: Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        threshold: float, optional
            Threshold for the calibration matrix. Default: 0.05.
        kernel_size: int, optional
            Kernel size for the calibration matrix. Default: 7.
        crop: float, optional
            Output eigenvalue cropping threshold. Default: 0.95.
        max_iter: int, optional
            Power method iterations. Default: 30.
        kspace_key: KspaceKey
            K-space key. Default KspaceKey.masked_kspace.
        """
        self.backward_operator = backward_operator
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.crop = crop
        self.max_iter = max_iter
        self.kspace_key = kspace_key
        super().__init__()

    @torch.no_grad()
    def calculate_sensitivity_map(self, acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
        """Calculates sensitivity map given as input the `acs_mask` and the `k-space`.

        Parameters
        ----------
        acs_mask : torch.Tensor
            Autocalibration mask.
        kspace : torch.Tensor
            K-space.

        Returns
        -------
        sensitivity_map : torch.Tensor
        """
        # pylint: disable=too-many-locals
        kshape = kspace.shape
        # print("K-Shape", kshape)
        ndim = kspace.ndim - 2
        spatial_size = kspace.shape[1:-1]

        # Used in case the k-space is padded (e.g. for batches)
        non_padded_dim = kspace.clone().sum(dim=tuple(range(1, kspace.ndim))).bool()

        num_coils = non_padded_dim.sum()
        acs_kspace_cropped = view_as_complex(crop_to_acs(acs_mask.squeeze(), kspace[non_padded_dim]))
        # print('acs_kspace_cropped', acs_kspace_cropped.shape)

        # Get calibration matrix.
        calibration_matrix = (
            nn.functional.unfold(acs_kspace_cropped.unsqueeze(1), kernel_size=self.kernel_size, stride=1)
            .transpose(1, 2)
            .to(acs_kspace_cropped.device)
            .reshape(
                num_coils,
                *(np.array(acs_kspace_cropped.shape[1:3]) - self.kernel_size + 1),
                *([self.kernel_size] * ndim),
            )
        )
        # print("calibration_matrix", calibration_matrix.shape)
        calibration_matrix = calibration_matrix.reshape(num_coils, -1, self.kernel_size**ndim)
        # print("calibration_matrix", calibration_matrix.shape)
        calibration_matrix = calibration_matrix.permute(1, 0, 2)
        # print("calibration_matrix", calibration_matrix.shape)
        calibration_matrix = calibration_matrix.reshape(-1, num_coils * self.kernel_size**ndim)
        # print("calibration_matrix", calibration_matrix.shape)
        # Perform SVD on calibration matrix
        
        _, s, vh = torch.linalg.svd(calibration_matrix, full_matrices=True)
        # print("S", s.shape)
        # print("VH", vh.shape)
        vh = vh[s > (self.threshold * s.max()), :]

        # Get kernels
        num_kernels = vh.shape[0]
        kernels = vh.reshape([num_kernels, num_coils] + [self.kernel_size] * ndim)

        # Get covariance matrix in image domain
        covariance = torch.zeros(
            spatial_size[::-1] + (num_coils, num_coils),
            dtype=kernels.dtype,
            device=kernels.device,
        )
        for kernel in kernels:
            pad_h, pad_w = (
                spatial_size[0] - self.kernel_size,
                spatial_size[1] - self.kernel_size,
            )
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            kernel_padded = torch.nn.functional.pad(kernel, pad)

            img_kernel = self.backward_operator(kernel_padded, dim=(1, 2), complex_input=False)
            aH = img_kernel.permute(*torch.arange(img_kernel.ndim - 1, -1, -1)).unsqueeze(-1)
            a = aH.transpose(-1, -2).conj()
            covariance += aH @ a

        covariance = covariance * (np.prod(spatial_size) / self.kernel_size**ndim)
        sensitivity_map = torch.ones(
            (*spatial_size[::-1], num_coils, 1),
            dtype=kernels.dtype,
            device=kernels.device,
        )

        def forward(x):
            return covariance @ x

        def normalize(x):
            return (x.abs() ** 2).sum(dim=-2, keepdims=True) ** 0.5

        power_method = MaximumEigenvaluePowerMethod(forward, max_iter=self.max_iter, norm_func=normalize)
        power_method.fit(x=sensitivity_map)

        temp_sensitivity_map = power_method.x.squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map.permute(
            *torch.arange(temp_sensitivity_map.ndim - 1, -1, -1)
        ).squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map * temp_sensitivity_map.conj() / temp_sensitivity_map.abs()

        max_eig = power_method.max_eig.squeeze()
        max_eig = max_eig.permute(*torch.arange(max_eig.ndim - 1, -1, -1))
        temp_sensitivity_map = temp_sensitivity_map * (max_eig > self.crop)

        sensitivity_map = torch.zeros_like(kspace, device=kspace.device, dtype=kspace.dtype)
        sensitivity_map[non_padded_dim] = view_as_real(temp_sensitivity_map)
        return sensitivity_map

    def forward(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Forward method of :class:`EspiritCalibration`.

        Parameters
        ----------
        sample: Dict[str, Any]
             Contains key `kspace_key`.

        Returns
        -------
        sample: Dict[str, Any]
             Contains key 'sampling_mask'.
        """
        acs_mask = sample["acs_mask"]
        kspace = sample[self.kspace_key]
        sensitivity_map = torch.stack(
            [self.calculate_sensitivity_map(acs_mask[_], kspace[_]) for _ in range(kspace.shape[0])],
            dim=0,
        ).to(kspace.device)

        return sensitivity_map
    
class Algorithm(ABC):
    """Base class for implementing mathematical optimization algorithms."""

    def __init__(self, max_iter: int = 30):
        self.max_iter = max_iter
        self.iter = 0

    @abstractmethod
    def _update(self):
        """Abstract method for updating the algorithm's parameters."""
        raise NotImplementedError

    @abstractmethod
    def _fit(self, *args, **kwargs):
        """Abstract method for fitting the algorithm.

        Parameters
        ----------
        *args : tuple
            Tuple of arguments.
        **kwargs : dict
            Keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def _done(self) -> bool:
        """Abstract method for checking if the algorithm has ran for `max_iter`.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def update(self) -> None:
        """Update the algorithm's parameters and increment the iteration count."""
        self._update()
        self.iter += 1

    def done(self) -> bool:
        """Check if the algorithm has converged.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self._done()

    def fit(self, *args, **kwargs) -> None:
        """Fit the algorithm.

        Parameters
        ----------
        *args : tuple
            Tuple of arguments for `_fit` method.
        **kwargs : dict
            Keyword arguments for `_fit` method.
        """
        self._fit(*args, **kwargs)
        while not self.done():
            self.update()


class MaximumEigenvaluePowerMethod(Algorithm):
    """A class for solving the maximum eigenvalue problem using the Power Method."""

    def __init__(
        self,
        forward_operator: Callable,
        norm_func: Optional[Callable] = None,
        max_iter: int = 30,
    ):
        """Inits :class:`MaximumEigenvaluePowerMethod`.

        Parameters
        ----------
        forward_operator : Callable
            The forward operator for the problem.
        norm_func : Callable, optional
            An optional function for normalizing the eigenvector. Default: None.
        max_iter : int, optional
            Maximum number of iterations to run the algorithm. Default: 30.
        """
        self.forward_operator = forward_operator
        self.norm_func = norm_func
        super().__init__(max_iter)

    def _update(self) -> None:
        """Perform a single update step of the algorithm.

        Updates maximum eigenvalue guess and corresponding eigenvector.
        """
        y = self.forward_operator(self.x)
        if self.norm_func is None:
            self.max_eig = (y * self.x.conj()).sum() / (self.x * self.x.conj()).sum()
        else:
            self.max_eig = self.norm_func(y)
        self.x = y / self.max_eig

    def _done(self) -> bool:
        """Check if the algorithm is done.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self.iter >= self.max_iter

    def _fit(self, x: torch.Tensor) -> None:
        """Sets initial maximum eigenvector guess.

        Parameters
        ----------
        x : torch.Tensor
            Initial guess for the eigenvector.
        """
        # pylint: disable=arguments-differ
        self.x = x
    

def get_acs_mask(mask: np.ndarray, half_bandwidth: int = 8, is_radial=False) -> np.ndarray:
    """
    Get auto-calibration mask from the true Cartesian under-sampling mask along ky.
    :param mask: Under-sampling mask of shape (..., kx, ky). DC component should be already shifted to k-space center.
    :param half_bandwidth: DC ± half_bandwidth is always sampled, this band will be used for calibration.
    :return: ACS mask of shape (..., kx, ky).
    """
    ky = mask.shape[-1]
    kx = mask.shape[-2]
    dc_ky_ind = ky // 2
    dc_kx_ind = kx // 2
    kx_slicer = slice(0, kx, 1)
    ky_slicer = slice(dc_ky_ind - half_bandwidth, dc_ky_ind + half_bandwidth, 1)
    if is_radial:
        acs_mask = np.zeros_like(mask)
        acs_mask[..., kx_slicer, ky_slicer] = 1.
        return acs_mask
    if is_radial:
        kx_slicer = slice(dc_kx_ind - half_bandwidth, dc_kx_ind + half_bandwidth, 1)
    assert np.all(mask[..., kx_slicer, ky_slicer] == 1),\
        "Central lines around ky-DC not fully sampled!"
    acs_mask = np.zeros_like(mask)
    acs_mask[..., kx_slicer, ky_slicer] = 1.
    return acs_mask

@torch.no_grad()
def estimate_sensitivity_map(k_space: np.ndarray, mask: np.ndarray, is_radial: bool = False, th: float=0.02) -> np.ndarray:
    """
    Estimation of sensitivity map.
    :param k_space: (kx, ky, nc, nb)
    :param mask: (kx, ky)
    :return: Sensitivity map of shape (kx, ky, nc, nb) complex.
    """
    # print("KSPACE", k_space.shape)
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cuda'
    k_space = np.transpose(k_space, (3, 2, 0, 1))
    Uacs = get_acs_mask(mask, half_bandwidth=8, is_radial=is_radial)
    # plt.imshow(Uacs)
    # print("GOT ACS MASK:", Uacs.shape)
    k_space = torch.from_numpy(k_space).to(device)
    Uacs = torch.from_numpy(Uacs).to(device)

    def backward_operator(*args, **kwargs):
        kwargs['normalized'] = True
        return ifft2(*args, **kwargs)

    sensitivity = []
    for b in range(k_space.shape[0]):
        sensitivity_estimator = EspiritCalibration(
            threshold=th,
            max_iter=30,
            crop=0.9,
            backward_operator=backward_operator
        )
        y = view_as_real(k_space[b, ...])
        S = sensitivity_estimator.calculate_sensitivity_map(Uacs, y)
        S = view_as_complex(S)
        sensitivity.append(S.detach().cpu().numpy())
    sensitivity = np.array(sensitivity)
    sensitivity = np.transpose(sensitivity, (2, 3, 1, 0))
    return torch.from_numpy(sensitivity)