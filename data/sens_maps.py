import numpy as np
import torch

def fft(X: np.ndarray, ax: list) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
def ifft(X: np.ndarray, ax: list) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

@torch.no_grad()
def espirit(X: torch.Tensor, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel 
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the 
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to('cuda:4')
    device = X.device
    # print(device)
    sx = X.shape[0]
    sy = X.shape[1]
    sz = X.shape[2]
    nc = X.shape[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)

    # Extract calibration region.    
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].to(torch.complex64)


    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = torch.zeros(((r-k+1)**p, k**p * nc), dtype=torch.complex64, device=device)


    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
      for ydx in range(max(1, C.shape[1] - k + 1)):
        for zdx in range(max(1, C.shape[2] - k + 1)):
          block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :]
          A[idx, :] = block.flatten()
          idx = idx + 1


    # Take the Singular Value Decomposition.
    U, S, VH = torch.linalg.svd(A, full_matrices=False)
    V = VH.conj().T
    # print("GENERATED SVD")
    # Select kernels.
    n = min(torch.sum(S >= t * S[0]), 32)

    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)



    # Reshape into k-space kernel, flips it and takes the conjugate
    _1, _2, _3, _4 = X.shape
    kernels = torch.zeros((_1, _2, _3, _4, n), dtype=torch.complex64, device=device) #.astype(np.complex64)
    kerdims = ((sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc)
    
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = torch.reshape(V[:, idx].clone(), kerdims)

    # print("GENERATED KERNELS")

    axes = (0, 1, 2)
    _1, _2, _3, _4 = X.shape
    kerimgs = torch.zeros((_1, _2, _3, _4, n), dtype=torch.complex64, device=device) #.astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = torch.flip(kernels, (0,1,2))[..., jdx, idx].conj()
            kerimgs[:,:,:,jdx,idx] = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ker), dim=axes, norm='ortho'))  * np.sqrt(sx * sy * sz)/np.sqrt(k**p)
    # print("GENERATED IUFFT", kerimgs.shape)
    kerimgs_flat = kerimgs.reshape(-1, nc, kerimgs.shape[-1])
    # Perform batched SVD
    u, s, vh = torch.linalg.svd(kerimgs_flat, full_matrices=False)
    # print("GENERATED SVD AGAIN")
    # Apply thresholding
    mask = (s**2 > c).unsqueeze(1)
    maps_flat = u * mask
    # Reshape back to original dimensions
    maps = maps_flat.reshape(sx, sy, sz, nc, nc).cpu()
    torch.cuda.empty_cache()
    return maps.squeeze().cpu().numpy()[..., 0]
    