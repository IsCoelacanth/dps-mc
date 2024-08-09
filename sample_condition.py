from functools import partial
import os
import argparse
import yaml

import torch
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
    sample_fn = partial(
        sampler.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        operator=operator,
    )

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config["operator"]["name"])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ["input", "recon", "progress", "label"]:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config["data"]
    dataset = get_dataset(**data_config)
    loader = get_dataloader(dataset, batch_size=4, num_workers=0, train=False)

    # Do Inference
    print("TOTAL BATCHES: ", len(loader))
    for i, (ref_img, guide) in enumerate(loader):
        logger.info(f"Inference for image {i}")
        ref_img = ref_img.to(device)

        for k in guide:
            guide[k] = guide[k].to(device)

        y_n = operator.forward(
            ref_img, mask=guide["mask"], sense_maps=guide["sense_maps"]
        )
        x_start = operator.At(y_n, guide["sense_maps"], ref_img.shape[-2:])
        # y_n = noiser(y)

        # start recon from the UC data samples
        x_start = x_start.clone().detach()

        # Sampling
        np.save("start.npy", x_start.detach().cpu().numpy())
        np.save("gt.npy", ref_img.detach().cpu().numpy())
        sample = sample_fn(
            x_start=x_start,
            measurement=y_n,
            record=False,
            save_root=out_path,
            guidance=guide,
        )
        # exit()
        i = 0
        for sl, fr in zip(guide["slice_no"], guide["frame_no"]):
            np.save(
                f"{sl}__{fr}.npy",
                {
                    "gt": ref_img[i].squeeze().detach().cpu().numpy(),
                    "pr": sample[i].detach().squeeze().cpu().numpy(),
                },
                allow_pickle=True,
            )
        # for i, fn in enumerate(file_name):
        #     result = {
        #             'undersampled_input': inputs[i].detach().cpu().numpy(),
        #             'ground_truth': gt[i].detach().cpu().numpy(),
        #             'recon': sample[i].detach().cpu().numpy(),
        #             'kspace': kspace[i].detach().cpu().numpy(),
        #             }
        #     np.save(os.path.join(out_path, fn), result, allow_pickle=True) # type: ignore


if __name__ == "__main__":
    main()
