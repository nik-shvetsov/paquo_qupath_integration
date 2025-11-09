ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS builder

ENV LANG=C.UTF-8 \
LC_ALL=C.UTF-8 \
PYTHON_VERSION=3.12 \
DEBIAN_FRONTEND=noninteractive \
NVIDIA_VISIBLE_DEVICES=all \
NVIDIA_DRIVER_CAPABILITIES=compute,utility \
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PATH=/opt/conda/bin:$PATH

RUN apt-get update && \
apt-get install -y --no-install-recommends \
build-essential \
ca-certificates \
curl \
xz-utils && \
apt-get upgrade -y && \
mkdir -p /workspace/paquo/models /workspace/paquo/scripts workspace/projects /workspace/data

RUN curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
chmod +x /tmp/miniconda.sh && \
/tmp/miniconda.sh -b -p /opt/conda && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
conda install -c conda-forge -y python=${PYTHON_VERSION} pip openslide pyvips && \
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 && \
pip install scipy polars numba shapely openslide-python scikit-image opencv-python paquo && \
conda clean -ya

COPY cuda_mobilevit4.pt2 /workspace/paquo/models/cuda_mobilevit4.pt2
COPY scripts /workspace/paquo/scripts

FROM ${BASE_IMAGE} AS runtime

ENV LANG=C.UTF-8 \
LC_ALL=C.UTF-8 \
NVIDIA_VISIBLE_DEVICES=all \
NVIDIA_DRIVER_CAPABILITIES=compute,utility \
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PATH=/opt/conda/bin:$PATH \
QUPATH_VER=0.5.1

RUN apt-get update && \
apt-get install -y --no-install-recommends \
ca-certificates \
default-jre && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/src/*

COPY --from=builder /workspace/ /workspace/
COPY --from=builder /opt/conda/ /opt/conda/

RUN paquo get_qupath --install-path /workspace/ ${QUPATH_VER} && \
mv /workspace/QuPath-${QUPATH_VER} /workspace/QuPath

ENV PAQUO_QUPATH_DIR=/workspace/QuPath \
PAQUO_QUPATH_SEARCH_DIRS=/workspace/

WORKDIR /workspace
USER root

CMD ["/workspace/QuPath/bin/QuPath"]
