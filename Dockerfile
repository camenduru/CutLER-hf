FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:${PATH}
WORKDIR ${HOME}/app

RUN curl https://pyenv.run | bash
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}
ENV PYTHON_VERSION=3.10.9
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir -U torch==1.13.1 torchvision==0.14.1
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/detectron2.git@58e472e \
    git+https://github.com/cocodataset/panopticapi.git@7bb4655 \
    git+https://github.com/mcordts/cityscapesScripts.git@8da5dd0
RUN pip install --no-cache-dir -U \
    numpy==1.23.5 \
    scikit-image==0.19.2 \
    opencv-python-headless==4.6.0.66 \
    colored==1.4.4
RUN pip install --no-cache-dir -U gradio==3.16.2

COPY --chown=1000 . ${HOME}/app
ENV PYTHONPATH=${HOME}/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces
CMD ["python", "app.py"]
