FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 AS builder

# install ninja for faster compile
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt -y install ffmpeg git libsm6 libxext6 ninja-build wget

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# create environment with python 3.10
RUN /opt/conda/bin/conda create -y -n myenv python=3.10

# install Pip requirements
WORKDIR /app
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
COPY requirements.txt /app
RUN pip install -r requirements.txt

# compile CUDA code
ARG TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0 7.5 8.6+PTX"
COPY --chown=algorithm:algorithm src/ src/
COPY --chown=algorithm:algorithm setup.py .
RUN pip install -e .


# reimport image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04 AS runner

# install libraries required for using the OpenCV python packages
RUN apt-get update  \
    && apt-get install ffmpeg libsm6 libxext6 -y  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy environment data including python
COPY --from=builder /opt/conda/envs/myenv/bin /opt/conda/envs/myenv/bin
COPY --from=builder /opt/conda/envs/myenv/lib /opt/conda/envs/myenv/lib
COPY --from=builder /app /app

# configure environment settings
ENV PATH=/usr/lib/x86_64-linux-gnu:/opt/conda/envs/myenv/bin:$PATH

# copy app data
WORKDIR /app
COPY . /app

# run python file
ENTRYPOINT ["python3", "process.py"]
