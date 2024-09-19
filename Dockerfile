FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# FROM nvcr.io/nvidia/pytorch:24.08-py3

RUN mkdir /code
WORKDIR /code

COPY  ./ /code/
COPY ./ema_0.9999_1000000.pt /code/model.pt

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip

RUN pip3 install -r /code/requires.txt
RUN 

ENTRYPOINT ["python", "sample_condition.py", "--model_config=configs/model_config.yaml", "--diffusion_config=configs/diffusion_config.yaml", "--task_config=configs/recon.yaml"]