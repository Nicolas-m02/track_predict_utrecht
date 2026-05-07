# Dockerfile for project

# inherit from python image
FROM python:3.11-slim

# add requirements file to container
ADD requirements.txt /requirements.txt
# run apt-get for installing torch.compile dependencies
#RUN apt-get update -qq
#RUN apt-get install -qq -y build-essential gcc
#RUN apt install build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg
# update to latest pip version 
RUN pip install -q -U pip
#RUN pip install -q -U manim

# check python version
RUN python --version

ENV PYTHONPATH "${PYTHONPATH}:/code/:/code/models"

# install the required python packages
RUN pip install -q -r /requirements.txt

# install set env variable for torch.compile
#ENV TRITON_PTXAS_PATH "/home/tbloecker/.local/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas"
