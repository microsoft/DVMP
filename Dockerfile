FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openssh-server  unzip curl \
    cmake gcc g++ \
    iputils-ping net-tools  iproute2  htop xauth \
    tmux wget vim git bzip2 ca-certificates \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
    
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /root/code

ENV envname pytorch
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n $envname python=3.6 && \
    conda activate $envname && \
    conda install -y pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch && \
    conda install -y tensorboard tqdm scipy scikit-learn && \
    git clone https://github.com/pytorch/fairseq && \
    cd fairseq && \
    git checkout dd74992d0d143155998e9ed4076826bcea80fb06 && \
    pip install  -e . && \
    chmod o+rwx -R /root && \
    chmod o+w -R /opt/conda/envs/${envname} && \
    cd .. && \
    conda install -y -c conda-forge rdkit=2020.09.5 && \
    conda install -y tensorflow=2.2.0 && \
    conda install -y -c conda-forge deepchem && \
    TORCH=1.8.0 && CUDA=cu102 && \
    pip install torch-scatter --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-sparse --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-cluster --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-spline-conv --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-geometric && \
    conda clean -ay && pip cache purge && \
    sed -i 's/conda activate base/conda activate '"$envname"'/g' /etc/profile

ENV MKL_THREADING_LAYER GNU
ENV PATH /opt/conda/envs/${envname}/bin:$PATH
EXPOSE 6006
RUN echo "export LANG=C.UTF-8" >> /etc/profile && \
    echo "export MKL_THREADING_LAYER=GNU" >> /etc/profile 