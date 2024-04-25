FROM nvidia/cuda:12.3.2-devel-ubuntu20.04

LABEL maintainer="Yucheng Shi; shiyucheng0923@gmail.com>"

# Set environment variables to make the build non-interactive and configure locales
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Etc/UTC

# Ensure tzdata installs without prompts
RUN echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y tzdata

# Update and install essential Ubuntu packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    vim \
    nano \
    zip \
    zlib1g-dev \
    unzip \
    pkg-config \
    libgl-dev \
    libblas-dev \
    liblapack-dev \
    python3-tk \
    python3-wheel \
    graphviz \
    libhdf5-dev \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    swig \
    apt-transport-https \
    lsb-release \
    libpng-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py \
    && rm get-pip.py \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python3


# Ensure system dependencies for matplotlib are installed
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    pkg-config \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install matplotlib and other Python packages
RUN pip install --no-cache-dir matplotlib

# Install essential Python packages
RUN pip3 --no-cache-dir install \
    "gymnasium" \
    "wandb" \
    "tqdm" \
    "tensorboard" \
    "pymoo" \
    "plotly" \
    "wheel" \
    "numpy" \
    "scipy" \
    "pandas" \
    "jupyter" \
    "scikit-learn" \
    "scikit-image" \
    "seaborn" \
    "graphviz" \
    "gpustat" \
    "h5py" \
    "gitpython" \
    "ptvsd" \
    "Pillow" \
    "opencv-python" \
    "cython==0.29.35" \
    "imageio==2.1.2" \
    "torch==1.13.1" \
    "torchvision==0.14.1" \
    "torchaudio==0.13.1" \
    "setuptools" \
    "gym==0.15.3" \
    "mujoco-py==2.1.2.14" \
    "ruamel.yaml<0.16.0" \
    "protobuf==3.17.2" \
    "mujoco==2.3.5"

ENV MUJOCO_DIR=/root/.mujoco

# MuJoCo setup
RUN mkdir -p $MUJOCO_DIR \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz \
    && tar -xzf $MUJOCO_DIR/mujoco210.tar.gz -C $MUJOCO_DIR \
    && rm $MUJOCO_DIR/mujoco210.tar.gz

WORKDIR /src

# MuJoCo key setup (assuming the key is passed as a build argument)
ARG MUJOCO_KEY=""
RUN echo "$MUJOCO_KEY" > /root/.mujoco/mjkey.txt
RUN cat /root/.mujoco/mjkey.txt
