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


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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

    with open(data_config["datapath"], "r") as f:
        val_files = json.load(f)
    print(
        "LOADED CONTRASTS: ", list(val_files.keys())
    )

    accelerations = data_config["accs"]

    sfe_mode = data_config["single_file_eval"]
    destination_dir = data_config['dest_dir']

    for contrast in val_files:
        if not os.path.exists(os.path.join(destination_dir, contrast)):
            os.makedirs(os.path.join(destination_dir, contrast), exist_ok=True)
        for acc in accelerations:
            if not os.path.exists(os.path.join(destination_dir, contrast, f'acc{acc}')):
                os.makedirs(os.path.join(destination_dir, contrast, f'acc{acc}'), exist_ok=True)
            for file in val_files[contrast]:
                # make mask-path for current file:
                file_mask = np.load(f'/home/anurag/Code/DiffuseRecon/acc{acc}-{contrast}-mask.npy')
                dataset = get_dataset(
                    name=data_config["name"],
                    root=file,
                    single_file_eval=sfe_mode,
                    mask_path=file_mask,
                )
                loader = get_dataloader(dataset, batch_size=4, num_workers=0, train=False)

                logger.info(f"Running Inference for file: {(contrast, file.split('/')[-2:])} @ acc = {acc} | num-batches: {len(loader)}")
                for i, (ref_img, guide) in enumerate(loader):
                    ref_img = ref_img.to(device)
                    for k in guide:
                        if k in ['shape', 'name', 'pid']:
                            # print(guide[k])
                            continue
                        guide[k] = guide[k].to(device)

                    y_n = operator.forward(
                        ref_img, mask=guide["mask"], sense_maps=guide["sense_maps"]
                    )
                    x_start = operator.At(y_n, guide["sense_maps"], ref_img.shape[-2:])
                    x_start = x_start.clone().detach()
                    
                    # create starting seed image
                    forw = forward_fn(x_start=x_start)

                    # MRI Recon
                    sample = sample_fn(
                        x_start=forw,
                        measurement=y_n,
                        record=False,
                        save_root=out_path,
                        guidance=guide,
                    )
                    ih, iw = guide['shape']
                    ih = ih[0].item()
                    iw = iw[0].item()
                    sample = interpolate(sample, size=(ih, iw), mode='nearest-exact')
                    gt_image = guide['gt_image']

                    # 
                    # exit()
                    i = 0
                    for name in guide['name']:
                        save_name = os.path.join(destination_dir, contrast, f"acc{acc}", name)
                        np.save(
                            save_name,
                            {
                                'in': x_start[i].squeeze().detach().cpu().numpy(),
                                "gt": gt_image[i].squeeze().detach().cpu().numpy(),
                                "pr": sample[i].detach().squeeze().cpu().numpy(),
                            },
                            allow_pickle=True,
                        )


if __name__ == "__main__":
    main()
