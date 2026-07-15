#!/usr/bin/env bash
# Source this before TensorFlow commands so pip/uv NVIDIA CUDA wheels are visible.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES="${PROJECT_ROOT}/.venv/lib/python3.11/site-packages"
CUDA_NVCC="${SITE_PACKAGES}/nvidia/cuda_nvcc"

TF_GPU_LIB_DIRS=(
  "${SITE_PACKAGES}/nvidia/cublas/lib"
  "${SITE_PACKAGES}/nvidia/cuda_cupti/lib"
  "${SITE_PACKAGES}/nvidia/cuda_nvrtc/lib"
  "${SITE_PACKAGES}/nvidia/cuda_runtime/lib"
  "${SITE_PACKAGES}/nvidia/cudnn/lib"
  "${SITE_PACKAGES}/nvidia/cufft/lib"
  "${SITE_PACKAGES}/nvidia/curand/lib"
  "${SITE_PACKAGES}/nvidia/cusolver/lib"
  "${SITE_PACKAGES}/nvidia/cusparse/lib"
  "${SITE_PACKAGES}/nvidia/nccl/lib"
  "${SITE_PACKAGES}/nvidia/nvjitlink/lib"
)

TF_GPU_LD_LIBRARY_PATH="$(IFS=:; echo "${TF_GPU_LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="${TF_GPU_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${CUDA_NVCC}/bin${PATH:+:${PATH}}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC}${XLA_FLAGS:+ ${XLA_FLAGS}}"

