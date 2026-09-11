FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ARG USERNAME=finx
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/envs/stablecoin/bin:${CONDA_DIR}/bin:${PATH}
ENV PYTHONPATH=/workspace/stablecoin/src
ENV PROJECT_DIR=/workspace/stablecoin
ENV MPLBACKEND=Agg
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        wget \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libxext6 \
        libsm6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME}

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    conda create -y -n stablecoin python=3.9.7 pip && \
    conda clean -afy

COPY src/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip wheel && \
    python -m pip install "setuptools<81" && \
    python -m pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip cache purge

RUN mkdir -p /workspace/stablecoin && \
    chown -R ${USERNAME}:${USERNAME} /workspace /opt/conda /home/${USERNAME}

USER ${USERNAME}
WORKDIR /workspace/stablecoin

CMD ["/bin/bash"]
