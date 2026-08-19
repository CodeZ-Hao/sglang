# DeepSeek-V4-Flash DSpark + KT native-MXFP4 installation guide

This guide installs and runs the paired Trilog SGLang and KTransformers
branches for `deepseek-ai/DeepSeek-V4-Flash-0731` with:

- DSpark speculative decoding;
- an RTX PRO 6000 Blackwell target GPU (SM120);
- optional DSpark draft execution on an RTX 4090 (SM89);
- native checkpoint MXFP4 expert weights on GPU and CPU;
- KT-Kernel AMX-BF16 acceleration for CPU-resident experts;
- CUDA 13.3 and a PyTorch CUDA 13 environment;
- static uniform or recorded-frequency expert placement.

No AMXINT4 conversion is used. The target and KT CPU backend both consume the
checkpoint's native packed E2M1 values and UE8M0 scales.

## Validated configuration

The commands in this guide target the following software combination:

| Component | Version or branch |
| --- | --- |
| Python | 3.11 |
| CUDA toolkit used by NVCC/JIT | 13.3 |
| PyTorch | 2.11.0 with CUDA 13.0 wheels |
| FlashInfer Python | 0.6.15.post1 |
| SGLang | `trilog-inc/sglang:codex/dspark-amx` |
| KTransformers | `trilog-inc/ktransformers:codex/dspark-amx-native` |
| Target GPU | RTX PRO 6000 Blackwell, compute capability 12.0 |
| Optional draft GPU | RTX 4090, compute capability 8.9 |
| CPU ISA | AMX-TILE and AMX-BF16 |

The validated machine used NVIDIA driver `610.43.02`. A different driver is
acceptable only if it supports the installed CUDA 13.x toolkit and both GPUs.

The implementation currently assumes `--tp 1`. The RTX 4090 is used only for
the DSpark draft; it is not a tensor-parallel target rank. Offloading the draft
does not move the target weights or target KV cache away from the RTX PRO 6000,
so high target-GPU memory usage remains normal.

## Repository relationship

These two branches must be used together:

- <https://github.com/trilog-inc/sglang/tree/codex/dspark-amx>
- <https://github.com/trilog-inc/ktransformers/tree/codex/dspark-amx-native>

SGLang owns model execution, DSpark, CUDA graphs, heterogeneous target/draft
device handling, expert placement, and the tuning tools. KTransformers provides
the native MXFP4 AMX/AVX CPU operator and checkpoint loader.

Do not install the KTransformers repository's bundled SGLang submodule. Clone
and install the Trilog SGLang branch separately as shown below.

## 1. Check the host

Run these before creating the environment:

```bash
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total \
  --format=csv,noheader

/usr/local/cuda-13.3/bin/nvcc --version

grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | \
  grep -E '^(amx_tile|amx_bf16|avx512_bf16)$'

numactl --hardware
```

The CPU check must show at least `amx_tile` and `amx_bf16`. The target GPU must
report compute capability `12.0`.

On Ubuntu or Debian, install the native build prerequisites once:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  curl \
  git \
  git-lfs \
  libhwloc-dev \
  libnuma-dev \
  numactl \
  pkg-config

git lfs install
```

## 2. Create the Conda environment

Python 3.11 is recommended. It matches the validated deployment and satisfies
KT-Kernel's `>=3.11` requirement.

```bash
conda create -n dsv4 python=3.11 -y
conda activate dsv4

conda install -c conda-forge -y \
  cmake \
  ninja \
  rust

python -m pip install --upgrade \
  pip \
  setuptools \
  wheel \
  packaging \
  pybind11
```

Every build and launch command below must be run with `dsv4` activated.

## 3. Select CUDA 13.3

Set these in every build or inference shell:

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Confirm that Conda is using the intended Python and toolkit:

```bash
which python
python --version
which nvcc
nvcc --version
```

Do not rely on an unversioned `/usr/local/cuda` symlink unless it points to the
13.3 installation.

## 4. Install PyTorch first

Install the CUDA 13.0 PyTorch build before either source tree. CUDA 13.3 NVCC is
used for compilation and JIT, while the tested PyTorch wheel reports CUDA 13.0.

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0' \
  'torchaudio==2.11.0' \
  torchvision
```

Verify it before continuing:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Expected output begins with:

```text
2.11.0+cu130 13.0
```

## 5. Clone the paired branches

The example layout matches the paths used in the launch commands:

```bash
mkdir -p /mnt/home_extend/llm/dsv4/dspark
cd /mnt/home_extend/llm/dsv4/dspark

git clone \
  --branch codex/dspark-amx \
  https://github.com/trilog-inc/sglang.git

git clone \
  --branch codex/dspark-amx-native \
  https://github.com/trilog-inc/ktransformers.git

cd /mnt/home_extend/llm/dsv4/dspark/ktransformers
git submodule update --init \
  third_party/pybind11 \
  third_party/llama.cpp
```

Confirm the branches:

```bash
git -C /mnt/home_extend/llm/dsv4/dspark/sglang \
  status --short --branch

git -C /mnt/home_extend/llm/dsv4/dspark/ktransformers \
  status --short --branch
```

## 6. Install SGLang

Install SGLang before KT-Kernel so that the final native KT extension is built
against the PyTorch version used by the server:

```bash
cd /mnt/home_extend/llm/dsv4/dspark/sglang
python -m pip install -e './python'
```

The editable install makes Python-only branch updates effective immediately
after `git pull`. Re-run the command if `python/pyproject.toml` changes.

Check the important versions:

```bash
python -c '
import flashinfer
import torch
import transformers
print("torch", torch.__version__, "torch CUDA", torch.version.cuda)
print("flashinfer", flashinfer.__version__)
print("transformers", transformers.__version__)
'
```

For this branch the expected FlashInfer version is `0.6.15.post1`.

## 7. Build KT-Kernel for native MXFP4 and AMX

Build on the inference server itself. `NATIVE` deliberately optimizes the
binary for that CPU and may make it unusable on older processors.

For the RTX PRO 6000, RTX 4090, and RTX 3090 machine, compile SM120, SM89, and
SM86 support:

```bash
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_ENABLE_AVX512=ON
export CPUINFER_ENABLE_AVX512_BF16=ON
export CPUINFER_USE_CUDA=1
export CPUINFER_CUDA_ARCHS='86;89;120'
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=16
export CPUINFER_FORCE_REBUILD=1

cd /mnt/home_extend/llm/dsv4/dspark/ktransformers/kt-kernel
python -m pip install \
  --no-build-isolation \
  --no-deps \
  -v \
  .
```

`--no-deps` is intentional. The KT-Kernel package metadata on this branch
still pins an older standalone PyTorch version; allowing pip to resolve that
dependency would replace the SGLang/PyTorch 2.11 CUDA 13 environment. The
extension is compiled against the already installed PyTorch 2.11 headers and
libraries.

If the RTX 3090 will never be used by this environment, use
`CPUINFER_CUDA_ARCHS='89;120'` to shorten the build.

Verify that native AMX-BF16 support was compiled:

```bash
python -c '
from kt_kernel import kt_kernel_ext
print("HAS_AMX_BF16", bool(getattr(kt_kernel_ext.moe, "HAS_AMX_BF16", False)))
'

kt doctor
```

`HAS_AMX_BF16` must be `True`. Do not continue with `--kt-mxfp4-backend amx`
if it is false.

## 8. Download the model

The native DeepSeek-V4-Flash checkpoint is used directly; no CPU weight
conversion is needed.

```bash
python -m pip install --upgrade 'huggingface_hub[hf_xet]'

hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /mnt/home_extend/models/DeepSeek-V4-Flash-0731
```

Both `--model-path` and `--kt-weight-path` must point at the same complete
checkpoint directory.

## 9. Launch the validated two-GPU configuration

With `CUDA_VISIBLE_DEVICES=0,2`, physical GPU 0 becomes logical `cuda:0` and
physical GPU 2 becomes logical `cuda:1`. Therefore the draft-device argument
is `--speculative-draft-device 1`.

```bash
conda activate dsv4
cd /mnt/home_extend/llm/dsv4/dspark/sglang

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST='8.9;12.0+PTX'

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,2 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --tp 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --kt-weight-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --kt-method MXFP4 \
  --kt-mxfp4-backend amx \
  --kt-mxfp4-amx-min-tokens-per-expert 4 \
  --kt-num-gpu-experts 96 \
  --kt-expert-placement-strategy uniform \
  --init-expert-location trivial \
  --kt-cpuinfer 26 \
  --kt-threadpool-count 2 \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1 \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-backend-prefill disabled \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --served-model-name dsv4flash \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-mxfp4-prefill-slots auto \
  --kt-mxfp4-prefill-host-staging-experts 8 \
  --speculative-draft-device 1 \
  --speculative-moe-runner-backend marlin \
  --host 0.0.0.0 \
  --port 60000
```

Important details:

- Do not pass `--fp8-gemm-backend cutlass`. It is deprecated on this hardware.
- Keep the target MoE backend `flashinfer_mxfp4`.
- Keep the SM89 draft MoE backend `marlin`.
- Keep decode CUDA graphs `breakable` while KT CPU work is enabled.
- Keep prefill CUDA graphs disabled for this configuration.
- Do not pass a separate draft model path. The checkpoint contains the DSpark
  draft configuration.

The uniform-placement startup line should report exactly 96 experts in every
one of the 43 routed layers:

```text
KT GPU placement: strategy=uniform total=4128/11008 per_layer={0: 96, ... 42: 96}
```

## 10. Smoke-test the server

Wait for `Application startup complete`, then run:

```bash
curl -fsS http://127.0.0.1:60000/model_info

curl -fsS http://127.0.0.1:60000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4flash",
    "messages": [{"role": "user", "content": "Explain speculative decoding briefly."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Test a short request first, then a long prompt, then concurrent requests. The
first request can be slower while JIT modules and caches are prepared.

## 11. Capture expert routing frequency

Capture using safe uniform placement and a trivial expert-location map. Add
these to the uniform launch:

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/mnt/home_extend/llm/dsv4/expert-profile
```

```text
--expert-distribution-recorder-mode stat
--expert-distribution-recorder-buffer-size 1000
```

Do not specify `--kt-expert-frequency-file` while producing the baseline
capture. Once the server is warm, start recording:

```bash
curl -fsS -X POST \
  http://127.0.0.1:60000/start_expert_distribution_record
```

Run a representative production workload. The model's expert distribution can
depend on language, prompt domain, input length, decode length, and concurrency.
If long prefills dominate the capture, they will also dominate the placement.

Stop and dump the recording:

```bash
curl -fsS -X POST \
  http://127.0.0.1:60000/stop_expert_distribution_record

curl -fsS -X POST \
  http://127.0.0.1:60000/dump_expert_distribution_record

ls -1t /mnt/home_extend/llm/dsv4/expert-profile/expert_distribution_recorder_*.pt | \
  head -1
```

Inspect a capture before using it:

```bash
python - /mnt/home_extend/llm/dsv4/expert-profile/expert_distribution_recorder_TIMESTAMP.pt <<'PY'
import sys
import torch

data = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
counts = data["logical_count"]
if counts.ndim == 3:
    counts = counts.sum(0)

print("shape:", tuple(counts.shape))
print("total routes:", int(counts.sum()))
print("layer totals min/max:", int(counts.sum(1).min()), int(counts.sum(1).max()))
coverage = counts.topk(96, dim=1).values.sum() / counts.sum().clamp_min(1)
print("top-96 per-layer coverage:", f"{100 * float(coverage):.2f}%")
PY
```

For DeepSeek-V4-Flash-0731, the reduced tensor must have shape `(43, 256)` and
every layer total must be nonzero.

## 12. Launch with frequency placement

Start from the validated launch command and change only the placement section:

```text
--kt-num-gpu-experts 96
--kt-expert-placement-strategy frequency
--kt-expert-frequency-file /mnt/home_extend/llm/dsv4/expert-profile/expert_distribution_recorder_TIMESTAMP.pt
--init-expert-location trivial
```

Never pass the recorder file to `--init-expert-location`. That option belongs
to SGLang EPLB and remaps expert IDs. KT static placement needs logical IDs and
uses the separate `--kt-expert-frequency-file` input.

Expected startup output:

```text
KT GPU placement: strategy=frequency total=4128/11008 per_layer={0: 96, ... 42: 96}
KT frequency profile: file=... selected_routes=.../... coverage=...%
```

Every routed layer must still have exactly 96 GPU experts. Compare throughput,
TTFT, TPOT, accepted DSpark length, and output quality with the uniform baseline.
A capture from an unrepresentative workload can legitimately perform worse
than uniform even when placement is working correctly.

## 13. Run the configuration tuner

Stop the production server first; the tuner starts and stops its own servers.

Inspect the planned search without launching:

```bash
cd /mnt/home_extend/llm/dsv4/dspark/sglang

python scripts/tune_dsv4_dspark_kt.py \
  --model-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0,2 \
  --speculative-draft-device 1 \
  --nvidia-smi-gpu 0 \
  --profile balanced \
  --dry-run
```

Run a small smoke sweep:

```bash
python scripts/tune_dsv4_dspark_kt.py \
  --model-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0,2 \
  --speculative-draft-device 1 \
  --nvidia-smi-gpu 0 \
  --profile smoke \
  --max-configs 3 \
  --finalists 2 \
  --output-dir tuning-smoke
```

Run uniform and frequency placement in the balanced search:

```bash
python scripts/tune_dsv4_dspark_kt.py \
  --model-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /mnt/home_extend/models/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0,2 \
  --speculative-draft-device 1 \
  --nvidia-smi-gpu 0 \
  --profile balanced \
  --placement-strategies uniform,frequency \
  --expert-frequency-path /mnt/home_extend/llm/dsv4/expert-profile/expert_distribution_recorder_TIMESTAMP.pt \
  --cpuinfer-step 4 \
  --output-dir tuning-balanced
```

The tuner passes `--expert-frequency-path` to the server as
`--kt-expert-frequency-file` and explicitly keeps
`--init-expert-location trivial`. It tests concurrency and long contexts and
writes a resumable `results.jsonl` plus `launch_best.sh`.

See `scripts/README_dsv4_dspark_tuning.md` for the full workload and scoring
description.

## 14. Updating an existing installation

Update SGLang:

```bash
conda activate dsv4
cd /mnt/home_extend/llm/dsv4/dspark/sglang
git pull --ff-only origin codex/dspark-amx
python -m pip install -e './python'
```

Update KTransformers and rebuild only when its branch changed:

```bash
cd /mnt/home_extend/llm/dsv4/dspark/ktransformers
git pull --ff-only origin codex/dspark-amx-native
git submodule update --init third_party/pybind11 third_party/llama.cpp

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_ENABLE_AVX512=ON
export CPUINFER_ENABLE_AVX512_BF16=ON
export CPUINFER_USE_CUDA=1
export CPUINFER_CUDA_ARCHS='86;89;120'
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=16
export CPUINFER_FORCE_REBUILD=1

cd kt-kernel
python -m pip install --no-build-isolation --no-deps -v .
```

## Troubleshooting

### `no kernel image is available for execution on the device`

First verify that both branch heads are current, `nvcc --version` reports 13.3,
and PyTorch sees both logical devices:

```bash
CUDA_VISIBLE_DEVICES=0,2 python -c '
import torch
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
'
```

The SGLang branch isolates JIT kernels by CUDA architecture and prepares a
multi-architecture FlashInfer sampling module for SM120 + SM89. Using an older
branch or a cache compiled before the heterogeneous fixes can reproduce this
error.

### `--fp8-gemm-backend=cutlass is deprecated on this hardware`

Remove that option. Use `--moe-runner-backend flashinfer_mxfp4` for the target
and `--speculative-moe-runner-backend marlin` for the RTX 4090 draft.

### Frequency placement collapses performance

Check the command and logs. The profile must be supplied through
`--kt-expert-frequency-file`, while `--init-expert-location` must be `trivial`.
Every layer must show 96 GPU experts. Do not reuse a profile captured by the old
unsafe expert-remapping command; recapture it with this guide.

### Long prompts fail near the 4096-token prefill boundary

As a diagnostic, disable the layerwise GPU prefill path:

```text
--kt-gpu-prefill-token-threshold 0
```

If that works, retest thresholds `2048`, `4096`, and `8192` with the tuner. You
can also reduce `--chunked-prefill-size` to `2048` while isolating memory or
prefill staging problems.

### CUDA graph replay fails around KT expert execution

Confirm these production settings:

```text
--cuda-graph-backend-decode breakable
--cuda-graph-backend-prefill disabled
```

Also remove debug-only variables before benchmarking:

```bash
unset CUDA_LAUNCH_BLOCKING
unset TORCH_SHOW_CPP_STACKTRACES
unset SGLANG_BCG_DEBUG_REPLAY
unset SGLANG_BCG_DEBUG_BREAKS
```

### RTX PRO 6000 utilization looks low

Decode can be CPU-expert-bound even when the GPU has available compute. Measure
output throughput and TPOT rather than GPU utilization alone. Tune the number
of GPU experts, frequency coverage, `--kt-cpuinfer` in four-thread intervals,
AMX crossover, and concurrency with the provided tuner.

### Out of memory during startup or long-context tests

Try the following in order:

1. Reduce `--mem-fraction-static` from `0.9` to `0.86` or `0.8`.
2. Reduce `--kt-num-gpu-experts` from `96` to `80` or `64`.
3. Set `--kt-mxfp4-prefill-slots 1`.
4. Reduce `--chunked-prefill-size`.
5. Reduce long-context concurrency or the requested context length.

Fewer GPU experts save VRAM but move more MoE work to the CPU, so always
remeasure throughput after changing that value.

## Operational checklist

Before treating a configuration as production-ready, confirm:

- `torch` is `2.11.0+cu130` and reports CUDA `13.0`;
- `nvcc` is CUDA `13.3`;
- FlashInfer is `0.6.15.post1`;
- `HAS_AMX_BF16` is true;
- GPU 0 is SM120 and the optional draft GPU is SM89;
- the server uses `MXFP4`, not `AMXINT4`;
- the target uses `flashinfer_mxfp4` and the remote draft uses `marlin`;
- decode graphs are `breakable` and prefill graphs are disabled;
- all 43 layers have the requested GPU expert count;
- frequency profiles use `--kt-expert-frequency-file` only;
- short, long-context, and concurrent generation all succeed;
- production benchmarks run without launch-blocking or BCG debug variables.
