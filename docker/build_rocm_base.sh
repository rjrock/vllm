AITER_BRANCH="v0.1.17"
PIP_EXTRA_INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"
PREBUILD_KERNELS="1"
PYTHON_VERSION="3.14"
PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950"
ROCM_SDK_VERSION="7.14.0"

DOCKER_BUILDKIT=1 docker build \
    --build-arg AITER_BRANCH="${AITER_BRANCH}" \
    --build-arg PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}" \
    --build-arg PREBUILD_KERNELS="${PREBUILD_KERNELS}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
    --build-arg ROCM_SDK_VERSION="${ROCM_SDK_VERSION}" \
    --build-arg USE_SCCACHE=1 \
    -f Dockerfile.rocm_base \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

