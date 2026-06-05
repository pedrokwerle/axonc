#!/usr/bin/env bash
# Creates a unified conda env for onnx2c + sauria_gen
set -e

ENV_NAME="${1:-axonc}"

echo "=== Creating conda env: $ENV_NAME ==="
conda create -y -n "$ENV_NAME" python=3.11 \
    cmake \
    ninja \
    gcc=15 \
    gxx=15 \
    libprotobuf \
    -c conda-forge -c defaults

echo "=== Activating ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing pip packages ==="
pip install \
    onnx==1.20.1 \
    onnx-ir==0.1.14 \
    onnxconverter-common==1.16.0 \
    onnxruntime==1.23.2 \
    onnxscript==0.5.7 \
    torch==2.9.1 \
    torchvision==0.24.1 \
    triton==3.5.1 \
    numpy \
    scipy \
    ml-dtypes==0.5.4 \
    coloredlogs==15.0.1 \
    humanfriendly==10.0 \
    filelock==3.20.3 \
    flatbuffers==25.12.19 \
    fsspec==2026.1.0 \
    jinja2==3.1.6 \
    markupsafe==3.0.3 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    packaging==25.0 \
    pillow==12.1.0 \
    protobuf==6.33.4 \
    pyyaml==6.0.2 \
    requests==2.32.3 \
    sympy==1.14.0 \
    typing-extensions==4.15.0

echo "=== Done. Activate with: conda activate $ENV_NAME ==="
