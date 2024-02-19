ARG CUDA=11.1.1
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
      vim \
      tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py37_4.12.0-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      openmm=7.5.1 \
      cudatoolkit==${CUDA_VERSION} \
      pdbfixer \
      pip \
      python=3.7

RUN conda install -y -c bioconda hhsuite==3.3.0 kalign2 tensorboardX

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz \
    && gunzip -c openmpi-4.1.1.tar.gz | tar xf - \
    && cd openmpi-4.1.1 \
    && ./configure --prefix=/usr/local --with-cuda \
    && make all install \
    && cd ..

COPY . /app/deepfold
RUN rm -rf /app/deepfold/.git

WORKDIR /app/deepfold

# Apply OpenMM patch.
WORKDIR /opt/conda/lib/python3.7/site-packages
RUN patch -p0 < /app/deepfold/openmm.patch

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real
RUN ldconfig

# Install pip packages.
RUN pip3 install --upgrade pip \
    && pip3 install jax==0.3.25 \
      jaxlib==0.3.25+cuda11.cudnn805 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip3 install mpi4jax==0.3.2 \
    && pip3 install -r /app/deepfold/requirements.txt

WORKDIR /app/deepfold
ENTRYPOINT ["/bin/bash"]
