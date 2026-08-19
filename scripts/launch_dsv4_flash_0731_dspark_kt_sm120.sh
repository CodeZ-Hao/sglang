#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731: DSpark + native MXFP4 KT offload on one SM120 GPU.
set -euo pipefail

ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then
  shift
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash-0731}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-${MODEL_PATH}}"
KT_KERNEL_ROOT="${KT_KERNEL_ROOT:-}"
KT_CPU_THREADS="${KT_CPU_THREADS:-$(getconf _NPROCESSORS_ONLN)}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
KT_NUMA_NODES="${KT_NUMA_NODES:-}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-96}"
KT_AMX_MIN_TOKENS_PER_EXPERT="${KT_AMX_MIN_TOKENS_PER_EXPERT:-4}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-4096}"
KT_MXFP4_PREFILL_SLOTS="${KT_MXFP4_PREFILL_SLOTS:-auto}"
KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS="${KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.86}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
DSPARK_DRAFT_DEVICE="${DSPARK_DRAFT_DEVICE:-}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

fail() {
  echo "error: $*" >&2
  exit 1
}

check_host() {
  [[ -x "${CUDA_HOME}/bin/nvcc" ]] || fail "nvcc not found at ${CUDA_HOME}/bin/nvcc"

  local nvcc_release
  nvcc_release="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
  [[ "${nvcc_release}" == "13.3" ]] || fail "CUDA 13.3 is required; nvcc reports ${nvcc_release:-unknown}"

  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"
  local compute_caps
  compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')"
  grep -qx '12.0' <<<"${compute_caps}" || fail "an SM120 GPU is required; compute capabilities: ${compute_caps:-unknown}"

  [[ -r /proc/cpuinfo ]] || fail "/proc/cpuinfo is unavailable"
  grep -qm1 -w amx_tile /proc/cpuinfo || fail "CPU does not advertise AMX-TILE"
  grep -qm1 -w amx_bf16 /proc/cpuinfo || fail "CPU does not advertise AMX-BF16"

  echo "validated: CUDA ${nvcc_release}, SM120, AMX-TILE, AMX-BF16"
}

build_kt() {
  check_host
  [[ -n "${KT_KERNEL_ROOT}" ]] || fail "set KT_KERNEL_ROOT to the ktransformers/kt-kernel directory"
  [[ -f "${KT_KERNEL_ROOT}/setup.py" ]] || fail "${KT_KERNEL_ROOT}/setup.py does not exist"

  export CPUINFER_CPU_INSTRUCT=NATIVE
  export CPUINFER_ENABLE_AMX=ON
  export CPUINFER_ENABLE_AVX512=ON
  export CPUINFER_ENABLE_AVX512_BF16=ON
  export CPUINFER_USE_CUDA=1
  export CPUINFER_CUDA_ARCHS=120
  python3 -m pip install --no-build-isolation -v "${KT_KERNEL_ROOT}"
}

serve() {
  check_host

  local numa_args=()
  local draft_device_args=()
  if [[ -n "${KT_NUMA_NODES}" ]]; then
    read -r -a numa_nodes <<<"${KT_NUMA_NODES}"
    numa_args=(--kt-numa-nodes "${numa_nodes[@]}")
  fi
  if [[ -n "${DSPARK_DRAFT_DEVICE}" ]]; then
    draft_device_args=(
      --speculative-draft-device "${DSPARK_DRAFT_DEVICE}"
      --speculative-moe-runner-backend marlin
    )
  fi

  exec python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --tp 1 \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algorithm DSPARK \
    "${draft_device_args[@]}" \
    --kt-weight-path "${KT_WEIGHT_PATH}" \
    --kt-method MXFP4 \
    --kt-mxfp4-backend amx \
    --kt-mxfp4-amx-min-tokens-per-expert "${KT_AMX_MIN_TOKENS_PER_EXPERT}" \
    --kt-gpu-prefill-token-threshold "${KT_GPU_PREFILL_TOKEN_THRESHOLD}" \
    --kt-mxfp4-prefill-slots "${KT_MXFP4_PREFILL_SLOTS}" \
    --kt-mxfp4-prefill-host-staging-experts "${KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS}" \
    --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}" \
    --kt-expert-placement-strategy uniform \
    --kt-cpuinfer "${KT_CPU_THREADS}" \
    --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
    "${numa_args[@]}" \
    --disable-shared-experts-fusion \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --swa-full-tokens-ratio 0.1 \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
}

case "${ACTION}" in
  check) check_host ;;
  build-kt) build_kt ;;
  serve) serve "$@" ;;
  *) fail "usage: $0 {check|build-kt|serve} [additional sglang arguments]" ;;
esac
