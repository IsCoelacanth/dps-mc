from collections import defaultdict
from functools import partial
import json
import os
import argparse
import yaml

import torch
from torch.nn.functional import interpolate
import numpy as np


from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.logger import get_logger
import hdf5storage

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def savenumpy2mat(data, np_var, filepath):
    ''' 
    np_var: str, the name of the variable in the mat file.
    data: numpy, array to save.
    filepath: str, the path to save the mat file.
    # attention! hdftstorage save the array in a mat file that both h5py and scipy.io.loadmat can read.
    # but it will transpose the data array.
    # If you want to save the file in the same way as the original mat file, please first apply np.transpose(data) 
    '''
    savedict= {}
    savedict[np_var] = data
    hdf5storage.savemat(filepath, savedict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--diffusion_config", type=str)
    parser.add_argument("--task_config", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    # set all the seeds for repo
    torch.manual_seed(42)
    np.random.seed(42)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # device_str = 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(**measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    logger.info(
        f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}"
    )

    # Prepare conditioning method
    cond_config = task_config["conditioning"]
    cond_method = get_conditioning_method(
        cond_config["method"], operator, noiser, **cond_config["params"]
    )
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    forwarder = create_sampler(**diffusion_config, forward=True)
    sample_fn = partial(
        sampler.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        operator=operator,
    )
    forward_fn = partial(forwarder.q_sample_loop, model=model)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config["operator"]["name"])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "recon", "progress", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config["data"]
    data_path = data_config["input"]
    contrasts = sorted(os.listdir(data_path))
    print("LOADED CONTRASTS: ", contrasts)
    sfe_mode = True
    dest_path = data_config["output"]

    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    for contrast in contrasts:
        if not os.path.exists(os.path.join(dest_path, contrast)):
            os.makedirs(os.path.join(dest_path, contrast), exist_ok=True)
            os.makedirs(os.path.join(dest_path, contrast, 'TestSet'), exist_ok=True)
            os.makedirs(os.path.join(dest_path, contrast, 'TestSet', 'Task2'), exist_ok=True)

        try:
            pids = os.listdir(os.path.join(data_path, contrast, "TestSet", "UnderSample_Task2"))
        except Exception as e:
            print(f"got {e} for contrast {contrast}, skipping")
            continue
        print(f"FOUND {len(pids)} PIDS for {contrast}")
        for pid in pids:
            if not os.path.exists(os.path.join(dest_path, contrast, 'TestSet', 'Task2', pid)):
                os.makedirs(os.path.join(dest_path, contrast, 'TestSet', 'Task2', pid))
            files = os.listdir(
                os.path.join(data_path, contrast, "TestSet", "UnderSample_Task2", pid)
            )
            for file in sorted(files):
                filename = os.path.join(
                    data_path, contrast, "TestSet", "UnderSample_Task2", pid, file
                )
                mask_name = filename.replace("UnderSample_Task2", "Mask_Task2").replace(
                    "_kus_", "_mask_"
                )
                dataset = get_dataset(
                    name=data_config["name"],
                    root=filename,
                    single_file_eval=True,
                    mask_path=mask_name,
                )
                dataloader = get_dataloader(
                    dataset, batch_size=6, num_workers=0, train=False
                )
                logger.info(f"Running Inference for file: {filename}, {len(dataloader)}")
                file_result_dict = defaultdict(list)
                keys_for_result = []
                for ref_img, guide in dataloader:
                    ref_img = ref_img.to(device)
                    for k in guide:
                        if k in ["shape", "stds", "max"]:
                            continue
                        guide[k] = guide[k].to(device)

                    y_n = operator.forward(
                        ref_img, mask=guide["mask"], sense_maps=guide["sense_maps"]
                    )
                    x_start = operator.At(y_n, guide["sense_maps"], ref_img.shape[-2:])
                    x_start = x_start.clone().detach()
                    forw = forward_fn(x_start=x_start)

                    # MRI Recon
                    sample = sample_fn(
                        x_start=forw,
                        measurement=y_n,
                        record=False,
                        save_root=out_path,
                        guidance=guide,
                    )
                    ih, iw = guide["shape"]
                    ih = ih[0].item()
                    iw = iw[0].item()
                    sample = interpolate(sample, size=(ih, iw), mode="nearest-exact")
                    gt_image = guide["gt_image"]

                    i = 0
                    for sl, fr in zip(guide["slice_no"], guide["frame_no"]):
                        if sl not in keys_for_result:
                            keys_for_result.append(sl.item())
                        file_result_dict[sl.item()].append(
                            (
                                fr.item(),
                                {
                                    "gt": gt_image[i].squeeze().detach().cpu().numpy(),
                                    "pr": sample[i].detach().squeeze().cpu().numpy(),
                                    "in": x_start[i].detach().squeeze().cpu().numpy(),
                                    "stds": guide["stds"],
                                    "max": guide["max"],
                                    "csm": guide["sense_maps"][i]
                                    .detach()
                                    .cpu()
                                    .squeeze()
                                    .numpy(),
                                    "ksp": guide["kspace"][i]
                                    .detach()
                                    .cpu()
                                    .squeeze()
                                    .numpy(),
                                    "mask": guide["mask"][i]
                                    .detach()
                                    .cpu()
                                    .squeeze()
                                    .numpy(),
                                },
                            )
                        )
                        i += 1
                slices = []
                for s in keys_for_result:
                    frames_s = file_result_dict[s]
                    frames_pr = []
                    for frameid, frames in frames_s:
                        pr = frames['pr']
                        mx = frames['max'][0].item()
                        std = frames['stds']['std1'][0].item()
                        pr = pr * mx
                        f = pr[0] + 1j*pr[1]
                        f = f * std
                        frames_pr.append(np.abs(f)) # [h, w]
                    slices.append(np.array(frames_pr))
                slices = np.array(slices)
                slices = np.transpose(slices, (1,0,2,3)).T

                # dest_name = filename.replace(data_path, dest_path).replace('UnderSample_Task2', "Task2")
                dest_name = os.path.join(dest_path, contrast, 'TestSet', 'Task2', pid, file)
                print(dest_path)
                savenumpy2mat(slices, 'img4ranking', dest_name)

                # np.save('result.npy', file_result_dict, allow_pickle=True)
                # exit()


if __name__ == "__main__":
    main()
