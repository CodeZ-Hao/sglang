# DeepSeek-V4-Flash DSpark/KT performance tuning

`tune_dsv4_dspark_kt.py` searches restart-required DSpark, KTransformers,
CUDA-graph, scheduling, and DSV4 kernel settings, then stress-tests the best
configurations across concurrent requests and long contexts.

The tuner uses the active Python environment. It starts and stops only the
server process groups it creates, refuses to use an occupied port, removes the
debug synchronization variables used during bring-up, records failures/OOMs,
and resumes completed trials from `results.jsonl`.

## Recommended run order

Activate the same conda environment used for the working server and enter the
SGLang checkout:

```bash
conda activate dsv4
cd /mnt/home_extend/llm/dsv4/dspark/sglang
```

First inspect the detected CPU/NUMA layouts, configurations, and workloads
without launching anything:

```bash
python scripts/tune_dsv4_dspark_kt.py \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /path/to/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0 \
  --nvidia-smi-gpu 0 \
  --profile balanced \
  --dry-run
```

Run a small end-to-end validation before committing to the full sweep:

```bash
python scripts/tune_dsv4_dspark_kt.py \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /path/to/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0 \
  --nvidia-smi-gpu 0 \
  --profile smoke \
  --max-configs 3 \
  --finalists 2 \
  --output-dir tuning-smoke
```

Then run the balanced sweep. The default bounded OFAT search includes the
four-thread CPUInfer sweep, adds at most eight mixed configurations, uses a
hard ceiling of 128 server configurations, and stress-tests three finalists:

```bash
python scripts/tune_dsv4_dspark_kt.py \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /path/to/DeepSeek-V4-Flash-0731 \
  --cuda-visible-devices 0 \
  --nvidia-smi-gpu 0 \
  --profile balanced \
  --output-dir tuning-balanced
```

Run the same command again to resume after an interruption. Pass `--no-resume`
to rotate the old `results.jsonl` and start a fresh measurement set. Successful
trials are skipped and failed trials are retried by default; pass
`--skip-failed` only when known-bad configurations should remain skipped.

The generated `tuning-balanced/launch_best.sh` contains the selected settings
and explicitly unsets launch-blocking/BCG diagnostic variables before starting
the production server.

## What is measured

The search stage measures:

- 128-token input / 256-token decode at concurrency 1;
- short decode at concurrency 8;
- 2K-token mixed prefill/decode at concurrency 4;
- 16K-token context at concurrency 1.

The balanced finalist stage measures decode concurrency 1, 4, 8, 16, and 32,
plus 8K, 32K, 128K, and 256K contexts. Long-context concurrency is bounded by
`--context-token-budget` (300K input tokens by default) to avoid requesting
more simultaneous KV state than this one-GPU configuration can normally hold.
Use `--profile exhaustive` to test every requested concurrency at every context
length that fits that budget.

Each benchmark records output/request throughput, TTFT, TPOT/ITL, p95/p99
latency, DSpark accepted length, request errors, output hashes, and sampled GPU
utilization, memory, clocks, power, and temperature. A normalized geometric
score favors output throughput and TPOT for decode workloads, and TTFT for
long-context workloads. The baseline score is 100.

Output hashes that differ from the baseline are reported but do not fail a
configuration by default: changing CPU/GPU expert placement can alter a
borderline greedy token because accumulation order changes. Add
`--require-output-match` if exact greedy output identity is required.

## Search space

For every comma-separated option, the first value is the baseline. Defaults:

```text
--gpu-experts             96,80,64
--amx-min-tokens          4,0,2,8
--gpu-prefill-thresholds  4096,2048,8192,0
--mxfp4-prefill-slots     auto,1
--prefill-host-staging-experts 8,16,4
--dspark-block-sizes      7,3,5
--chunked-prefill-sizes   4096,2048,8192
--max-running-requests    48,16,32
--mem-fractions           0.86
--placement-strategies    uniform,front-loading
--fuse-mhc-post-pre       false,true
--dspark-multistream      true,false
--ragged-verify-modes     static
```

The layerwise prefill path streams native MXFP4 experts into a prepared Marlin
layer once a chunk reaches `--kt-gpu-prefill-token-threshold`. `auto` attempts
two prepared slots for layer-to-layer overlap and falls back to one slot when
VRAM is constrained. Unlike the upstream two-raw/two-prepared design, this
branch stages a small raw expert window, so each slot needs only a prepared
layer rather than a full raw plus prepared layer. Startup logs report the exact
prepared/raw/total allocation for the loaded model. Set the threshold to `0`
to measure the original hybrid CPU/GPU prefill path.

The automatic CPU search keeps all online logical CPUs as the baseline, then
tests `--kt-cpuinfer` at 4-thread intervals (`4,8,12,...`) on the same all-NUMA
layout. On a multi-NUMA host it also tests GPU-local physical/logical endpoint
layouts. Change or bound the interval sweep with:

```bash
--cpuinfer-step 4 --cpuinfer-min 4 --cpuinfer-max 128
```

`--cpuinfer-max` defaults to the number of online logical CPUs. The detected
thread counts are printed by `--dry-run`. Make sure `--max-configs` is large
enough to include the desired high thread counts; its default is 128.

Override the automatic sweep completely with repeatable
`LABEL:THREADS:NODE,NODE` values:

```bash
--cpu-layout all-physical:112:0,1 \
--cpu-layout gpu-local:56:0
```

The OFAT strategy measures each individual change and adds up to
`--mixed-configs 8` deterministic mixed configurations without exceeding
`--max-configs`. A Cartesian search is available, but should be bounded because
every candidate reloads the model:

```bash
--search-strategy cartesian --max-configs 48
```

To test compact ragged verification with a profiled SPS table:

```bash
--ragged-verify-modes static,compact \
--sps-table-path /path/to/dspark_sps.json \
--align-verify-to-graph-tier
```

Frequency-based GPU expert placement requires a recorded expert-frequency
tensor:

```bash
--placement-strategies uniform,frequency \
--expert-frequency-path /path/to/expert_counts.pt
```

The tuner passes this file through `--kt-expert-frequency-file` and keeps
`--init-expert-location trivial`. Do not pass a recorder output through
`--init-expert-location`; that option enables SGLang expert remapping and is
incompatible with KT's compact logical expert maps.

Additional `launch_server` arguments must come last, after `--`:

```bash
python scripts/tune_dsv4_dspark_kt.py [tuner options] -- \
  --some-additional-server-option value
```

## Results

The output directory contains:

- `system_info.json`: CPU topology, GPU topology, driver, CUDA/PyTorch, and
  FlashInfer versions;
- `plan.json`: the exact candidate and workload matrix;
- `results.jsonl`: resumable raw trial records;
- `runs/`: server logs, benchmark logs, commands, raw JSONL, and telemetry;
- `ranking_search.csv`: ranking used to choose finalists;
- `ranking_stress.csv`: final ranking across concurrency and long contexts;
- `best_config.json`: selected config, environment, command, and score;
- `launch_best.sh`: directly executable production launch command.

Do not compare runs while `CUDA_LAUNCH_BLOCKING`, `SGLANG_BCG_DEBUG_REPLAY`, or
`SGLANG_BCG_DEBUG_BREAKS` is active. The tuner strips these variables from each
child server and benchmark process.
