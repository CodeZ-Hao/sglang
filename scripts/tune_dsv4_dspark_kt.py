#!/usr/bin/env python3
"""Restart-aware tuner for DSV4 DSpark + KT native MXFP4 inference.

The target model is expensive to load, so the default search is deliberately
not a full Cartesian product.  It evaluates the baseline, changes each knob
independently, adds a bounded set of mixed configurations, and then runs a
broader concurrency/long-context matrix on the finalists.  Use
``--search-strategy cartesian`` only when the resulting number of restarts is
acceptable.

This script intentionally uses only the Python standard library.  It launches
the installed SGLang from the active environment and invokes the branch's
``sglang.benchmark.serving`` entry point for measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEBUG_ENV_VARS = (
    "SGLANG_BCG_DEBUG_BREAKS",
    "SGLANG_BCG_DEBUG_REPLAY",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_SHOW_CPP_STACKTRACES",
)

FATAL_LOG_PATTERNS = (
    "scheduler hit an exception",
    "illegal memory access",
    "cuda error",
    "out of memory",
    "segmentation fault",
    "received sigquit from a child process",
)


@dataclass(frozen=True)
class CpuLayout:
    label: str
    cpuinfer_threads: int
    threadpool_count: int
    numa_nodes: tuple[int, ...]


@dataclass(frozen=True)
class ServerConfig:
    label: str
    gpu_experts: int
    amx_min_tokens: int
    gpu_prefill_threshold: int
    mxfp4_prefill_slots: str
    prefill_host_staging_experts: int
    dspark_block_size: int
    chunked_prefill_size: int
    max_running_requests: int
    mem_fraction_static: float
    placement: str
    fuse_mhc_post_pre: bool
    dspark_multistream: bool
    ragged_verify_mode: str
    cpu_layout: CpuLayout

    def identity_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("label", None)
        return data

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            self.identity_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int
    output_len: int
    concurrency: int
    num_prompts: int
    weight: float = 1.0
    latency_focus: bool = False

    @property
    def workload_id(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass
class TrialResult:
    config_id: str
    config_label: str
    stage: str
    workload: str
    workload_id: str
    status: str
    elapsed_s: float
    config: dict[str, Any]
    workload_config: dict[str, Any]
    metrics: dict[str, Any]
    telemetry: dict[str, Any]
    output_hash: str | None = None
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str, cast: Any = str) -> list[Any]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_bool_csv(value: str) -> list[bool]:
    result: list[bool] = []
    for item in parse_csv(value):
        normalized = item.lower()
        if normalized in ("1", "true", "yes", "on"):
            result.append(True)
        elif normalized in ("0", "false", "no", "off"):
            result.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid boolean value: {item!r}")
    return result


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def detect_cpu_topology() -> dict[str, Any]:
    node_root = Path("/sys/devices/system/node")
    nodes: dict[int, list[int]] = {}
    for node_dir in sorted(node_root.glob("node[0-9]*")):
        match = re.fullmatch(r"node(\d+)", node_dir.name)
        if not match:
            continue
        try:
            nodes[int(match.group(1))] = parse_cpu_list(
                (node_dir / "cpulist").read_text()
            )
        except OSError:
            continue

    logical_count = os.cpu_count() or 1
    if not nodes:
        nodes = {0: list(range(logical_count))}

    cpu_records: list[tuple[int, int, int, int]] = []
    for node, cpus in nodes.items():
        for cpu in cpus:
            topology_dir = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = read_int(topology_dir / "physical_package_id", node)
            core = read_int(topology_dir / "core_id", cpu)
            cpu_records.append((cpu, node, package, core))

    physical_cores = {(node, package, core) for _, node, package, core in cpu_records}
    physical_by_node = {
        node: len(
            {
                (package, core)
                for _, record_node, package, core in cpu_records
                if record_node == node
            }
        )
        for node in nodes
    }
    return {
        "nodes": nodes,
        "logical_cpus": len(cpu_records) or logical_count,
        "physical_cores": len(physical_cores) or logical_count,
        "physical_cores_by_node": physical_by_node,
    }


def detect_gpu_numa_node(nvidia_smi_gpu: str) -> int | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                nvidia_smi_gpu,
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    bus_id = output.splitlines()[0].strip().lower()
    numa_path = Path("/sys/bus/pci/devices") / bus_id / "numa_node"
    node = read_int(numa_path, -1)
    return node if node >= 0 else None


def auto_cpu_layouts(
    nvidia_smi_gpu: str,
    *,
    cpuinfer_step: int,
    cpuinfer_min: int,
    cpuinfer_max: int | None,
) -> list[CpuLayout]:
    topology = detect_cpu_topology()
    nodes: dict[int, list[int]] = topology["nodes"]
    node_ids = tuple(sorted(nodes))
    logical = int(topology["logical_cpus"])
    local_node = detect_gpu_numa_node(nvidia_smi_gpu)
    if local_node not in nodes:
        local_node = node_ids[0]

    threadpool_count = len(node_ids)
    upper = min(cpuinfer_max if cpuinfer_max is not None else logical, logical)
    lower = max(cpuinfer_min, threadpool_count)
    first_step = ((lower + cpuinfer_step - 1) // cpuinfer_step) * cpuinfer_step
    if first_step > upper and logical < lower:
        raise ValueError(
            f"CPUInfer lower bound {cpuinfer_min} exceeds the {logical} online CPUs"
        )

    # Keep the historical all-logical configuration as the baseline. Sweep
    # the same all-NUMA layout at four-thread intervals (or the requested
    # step), which isolates --kt-cpuinfer from NUMA placement changes.
    candidates = [CpuLayout("all_logical", logical, threadpool_count, node_ids)]
    label_width = max(2, len(str(logical)))
    for threads in range(first_step, upper + 1, cpuinfer_step):
        if threads == logical:
            continue
        candidates.append(
            CpuLayout(
                f"all_numa_cpuinfer_{threads:0{label_width}d}",
                threads,
                threadpool_count,
                node_ids,
            )
        )

    if len(node_ids) > 1:
        local_logical = len(nodes[local_node])
        local_physical = int(topology["physical_cores_by_node"][local_node])
        candidates.extend(
            [
                CpuLayout("gpu_numa_logical", local_logical, 1, (local_node,)),
                CpuLayout("gpu_numa_physical", local_physical, 1, (local_node,)),
            ]
        )
    return unique_preserving_order(candidates)


def parse_custom_cpu_layout(value: str) -> CpuLayout:
    # LABEL:THREADS:NODE,NODE
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "CPU layout must be LABEL:THREADS:NODE,NODE (for example all:112:0,1)"
        )
    label, threads_raw, nodes_raw = parts
    try:
        threads = int(threads_raw)
        nodes = tuple(parse_csv(nodes_raw, int))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if threads <= 0 or not nodes:
        raise argparse.ArgumentTypeError(
            "CPU layout needs positive threads and NUMA nodes"
        )
    return CpuLayout(label, threads, len(nodes), nodes)


def replace_config(config: ServerConfig, label: str, **changes: Any) -> ServerConfig:
    return replace(config, label=label, **changes)


def build_candidates(args: argparse.Namespace) -> list[ServerConfig]:
    cpu_layouts = args.cpu_layout or auto_cpu_layouts(
        args.nvidia_smi_gpu,
        cpuinfer_step=args.cpuinfer_step,
        cpuinfer_min=args.cpuinfer_min,
        cpuinfer_max=args.cpuinfer_max,
    )
    if not cpu_layouts:
        raise RuntimeError("no CPU layouts were detected or supplied")

    spaces: list[tuple[str, list[Any]]] = [
        ("gpu_experts", parse_csv(args.gpu_experts, int)),
        ("amx_min_tokens", parse_csv(args.amx_min_tokens, int)),
        ("gpu_prefill_threshold", parse_csv(args.gpu_prefill_thresholds, int)),
        ("mxfp4_prefill_slots", parse_csv(args.mxfp4_prefill_slots)),
        (
            "prefill_host_staging_experts",
            parse_csv(args.prefill_host_staging_experts, int),
        ),
        ("dspark_block_size", parse_csv(args.dspark_block_sizes, int)),
        ("chunked_prefill_size", parse_csv(args.chunked_prefill_sizes, int)),
        ("max_running_requests", parse_csv(args.max_running_requests, int)),
        ("mem_fraction_static", parse_csv(args.mem_fractions, float)),
        ("placement", parse_csv(args.placement_strategies)),
        ("fuse_mhc_post_pre", parse_bool_csv(args.fuse_mhc_post_pre)),
        ("dspark_multistream", parse_bool_csv(args.dspark_multistream)),
        ("ragged_verify_mode", parse_csv(args.ragged_verify_modes)),
        ("cpu_layout", cpu_layouts),
    ]
    for name, values in spaces:
        if not values:
            raise ValueError(f"empty search space for {name}")

    baseline_values = {name: values[0] for name, values in spaces}
    baseline = ServerConfig(label="baseline", **baseline_values)
    candidates: list[ServerConfig] = [baseline]

    if args.search_strategy == "cartesian":
        for values in itertools.product(*(values for _, values in spaces)):
            changes = dict(zip((name for name, _ in spaces), values))
            candidate = ServerConfig(label="cartesian", **changes)
            candidates.append(candidate)
    else:
        # One factor at a time makes the effect of each restart-required knob
        # interpretable and avoids an explosive model-reload count.
        for name, values in spaces:
            for value in values[1:]:
                display = value.label if isinstance(value, CpuLayout) else str(value)
                candidates.append(
                    replace_config(
                        baseline,
                        f"{name}={display}",
                        **{name: value},
                    )
                )

        # Use the remaining budget for deterministic mixed configurations so
        # interactions (for example gamma x concurrency x CPU layout) are not
        # completely missed by the OFAT pass.
        mixed_index = 0
        mixed_target = min(args.max_configs, len(candidates) + args.mixed_configs)
        while len(candidates) < mixed_target:
            changes: dict[str, Any] = {}
            for knob_index, (name, values) in enumerate(spaces):
                if len(values) == 1:
                    changes[name] = values[0]
                    continue
                offset = (mixed_index * (knob_index * 2 + 1) + knob_index) % len(values)
                changes[name] = values[offset]
            candidates.append(
                ServerConfig(label=f"mixed_{mixed_index + 1:02d}", **changes)
            )
            mixed_index += 1
            if mixed_index > args.max_configs * 4:
                break

    deduped: list[ServerConfig] = []
    identities: set[str] = set()
    for candidate in candidates:
        identity = json.dumps(candidate.identity_dict(), sort_keys=True)
        if identity in identities:
            continue
        identities.add(identity)
        deduped.append(candidate)
        if len(deduped) >= args.max_configs:
            break
    represented_cpu_layouts = {config.cpu_layout.label for config in deduped}
    omitted_cpu_layouts = [
        layout.label
        for layout in cpu_layouts
        if layout.label not in represented_cpu_layouts
    ]
    if omitted_cpu_layouts and args.search_strategy == "ofat":
        print(
            f"WARNING: --max-configs={args.max_configs} omits "
            f"{len(omitted_cpu_layouts)} CPUInfer/NUMA candidates. Raise "
            "--max-configs or lower --cpuinfer-max. First omitted: "
            + ",".join(omitted_cpu_layouts[:8]),
            file=sys.stderr,
        )
    return deduped[args.config_start : args.config_end]


def build_search_workloads(profile: str) -> list[Workload]:
    if profile == "smoke":
        return [
            Workload("decode_c1", 128, 64, 1, 1, 2.0),
            Workload("decode_c8", 128, 64, 8, 8, 2.0),
        ]
    return [
        Workload("decode_c1", 128, 256, 1, 3, 2.0),
        Workload("decode_c8", 128, 128, 8, 16, 2.0),
        Workload("mixed_2k_c4", 2048, 128, 4, 8, 1.0),
        Workload("context_16k_c1", 16384, 64, 1, 2, 1.0, True),
    ]


def build_stress_workloads(args: argparse.Namespace) -> list[Workload]:
    concurrencies = parse_csv(args.concurrencies, int)
    context_lengths = parse_csv(args.large_contexts, int)
    workloads: list[Workload] = []

    selected_concurrencies = concurrencies
    if args.profile == "smoke":
        selected_concurrencies = unique_preserving_order(
            [concurrencies[0], concurrencies[-1]]
        )
        context_lengths = context_lengths[:1]
    elif args.profile == "balanced" and len(concurrencies) > 5:
        indexes = (0, len(concurrencies) // 3, len(concurrencies) // 2, -2, -1)
        selected_concurrencies = unique_preserving_order(
            [concurrencies[index] for index in indexes]
        )

    for concurrency in selected_concurrencies:
        prompts = max(4, concurrency * 2)
        workloads.append(
            Workload(
                f"decode_c{concurrency}",
                128,
                args.decode_output_len,
                concurrency,
                prompts,
                2.0,
            )
        )

    for input_len in context_lengths:
        allowed = [
            concurrency
            for concurrency in concurrencies
            if input_len * concurrency <= args.context_token_budget
        ]
        if not allowed:
            allowed = [1]
        if args.profile == "exhaustive":
            selected = allowed
        elif args.profile == "smoke":
            selected = [1]
        else:
            selected = unique_preserving_order([1, allowed[-1]])
        for concurrency in selected:
            prompts = concurrency if input_len >= 65536 else max(2, concurrency * 2)
            workloads.append(
                Workload(
                    f"context_{input_len}_c{concurrency}",
                    input_len,
                    args.context_output_len,
                    concurrency,
                    prompts,
                    1.0,
                    True,
                )
            )
    return unique_preserving_order(workloads)


def config_environment(
    base: dict[str, str], config: ServerConfig, cuda_visible_devices: str
) -> dict[str, str]:
    env = dict(base)
    legacy_repo = env.pop("SGL_REPO", None)
    if legacy_repo is not None and "SGLANG_REPO" not in env:
        env["SGLANG_REPO"] = legacy_repo
    for name in DEBUG_ENV_VARS:
        env.pop(name, None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "KT_MXFP4_BACKEND": "amx",
            "KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT": str(config.amx_min_tokens),
            "SGLANG_OPT_USE_TILELANG_MHC_POST": "1",
            "SGLANG_OPT_FUSE_MHC_POST_PRE": ("1" if config.fuse_mhc_post_pre else "0"),
            "SGLANG_DSPARK_ENABLE_MULTI_STREAM": (
                "1" if config.dspark_multistream else "0"
            ),
            "SGLANG_RAGGED_VERIFY_MODE": config.ragged_verify_mode,
        }
    )
    return env


def build_server_command(args: argparse.Namespace, config: ServerConfig) -> list[str]:
    command = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        args.model_path,
        "--tp",
        "1",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-dspark-block-size",
        str(config.dspark_block_size),
        "--kt-weight-path",
        args.kt_weight_path,
        "--kt-method",
        "MXFP4",
        "--kt-mxfp4-backend",
        "amx",
        "--kt-mxfp4-amx-min-tokens-per-expert",
        str(config.amx_min_tokens),
        "--kt-gpu-prefill-token-threshold",
        str(config.gpu_prefill_threshold),
        "--kt-mxfp4-prefill-slots",
        config.mxfp4_prefill_slots,
        "--kt-mxfp4-prefill-host-staging-experts",
        str(config.prefill_host_staging_experts),
        "--kt-num-gpu-experts",
        str(config.gpu_experts),
        "--kt-expert-placement-strategy",
        config.placement,
        "--init-expert-location",
        "trivial",
        "--kt-cpuinfer",
        str(config.cpu_layout.cpuinfer_threads),
        "--kt-threadpool-count",
        str(config.cpu_layout.threadpool_count),
        "--kt-numa-nodes",
        *(str(node) for node in config.cpu_layout.numa_nodes),
        "--disable-shared-experts-fusion",
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--chunked-prefill-size",
        str(config.chunked_prefill_size),
        "--max-running-requests",
        str(config.max_running_requests),
        "--cuda-graph-backend-decode",
        "breakable",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--swa-full-tokens-ratio",
        str(args.swa_full_tokens_ratio),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.sps_table_path:
        command.extend(["--speculative-dspark-sps-table-path", args.sps_table_path])
    if args.speculative_draft_device:
        command.extend(
            [
                "--speculative-draft-device",
                args.speculative_draft_device,
                "--speculative-moe-runner-backend",
                "marlin",
            ]
        )
    if config.ragged_verify_mode == "compact" and args.align_verify_to_graph_tier:
        command.append("--speculative-dspark-align-verify-tokens-to-graph-tier")
    if args.expert_frequency_path and config.placement == "frequency":
        command.extend(["--kt-expert-frequency-file", args.expert_frequency_path])
    command.extend(args.extra_server_args)
    return command


def base_url(args: argparse.Namespace) -> str:
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    return f"http://{host}:{args.port}"


def http_json(url: str, timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def server_ready(url: str) -> bool:
    for endpoint in ("/model_info", "/health"):
        try:
            http_json(url + endpoint, timeout=3.0)
            return True
        except (OSError, ValueError, urllib.error.URLError):
            continue
    return False


def port_is_open(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((target, port), timeout=1.0):
            return True
    except OSError:
        return False


def tail_text(path: Path, max_bytes: int = 16000) -> str:
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            return file.read().decode(errors="replace")
    except OSError:
        return ""


def wait_for_server(
    process: subprocess.Popen[Any],
    url: str,
    timeout_s: float,
    log_path: Path,
) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited with status {return_code}\n{tail_text(log_path)}"
            )
        if server_ready(url):
            return time.monotonic() - start
        time.sleep(2.0)
    raise TimeoutError(
        f"server was not ready after {timeout_s:.0f}s\n{tail_text(log_path)}"
    )


def stop_process_group(process: subprocess.Popen[Any], timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


class GpuSampler:
    def __init__(self, gpu: str, interval_s: float = 1.0) -> None:
        self.gpu = gpu
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        return summarize_telemetry(self.samples)

    def _run(self) -> None:
        fields = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        self.gpu,
                        f"--query-gpu={fields}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.strip()
                values = [float(item.strip()) for item in output.split(",")]
                if len(values) == 6:
                    self.samples.append(
                        dict(
                            zip(
                                (
                                    "gpu_util_pct",
                                    "memory_mib",
                                    "power_w",
                                    "temperature_c",
                                    "sm_clock_mhz",
                                    "memory_clock_mhz",
                                ),
                                values,
                            )
                        )
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_s)


def summarize_telemetry(samples: Sequence[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    result: dict[str, Any] = {"samples": len(samples)}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[f"{key}_mean"] = statistics.fmean(values)
        result[f"{key}_max"] = max(values)
    return result


def build_benchmark_command(
    args: argparse.Namespace,
    workload: Workload,
    output_path: Path,
) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url(args),
        "--dataset-name",
        "random",
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--num-prompts",
        str(workload.num_prompts),
        "--random-input-len",
        str(workload.input_len),
        "--random-output-len",
        str(workload.output_len),
        "--random-range-ratio",
        "1",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(workload.concurrency),
        "--warmup-requests",
        str(args.warmup_requests),
        "--seed",
        str(args.seed),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--flush-cache",
        "--tokenize-prompt",
        "--output-details",
        "--disable-tqdm",
        "--output-file",
        str(output_path),
        "--tag",
        workload.name,
    ]


def hash_outputs(result: dict[str, Any]) -> str | None:
    outputs = result.get("generated_texts")
    if not isinstance(outputs, list):
        return None
    encoded = json.dumps(outputs, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_single_jsonl(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one benchmark record in {path}, got {len(records)}"
        )
    return records[0]


def validate_benchmark_result(
    result: dict[str, Any], workload: Workload
) -> tuple[bool, str | None]:
    if int(result.get("completed", 0)) != workload.num_prompts:
        return False, (
            f"completed {result.get('completed', 0)} of {workload.num_prompts} requests"
        )
    errors = [error for error in result.get("errors", []) if error]
    if errors:
        return False, errors[0][:2000]
    for metric in ("output_throughput", "median_tpot_ms", "p95_e2e_latency_ms"):
        value = result.get(metric)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            return False, f"missing or invalid metric {metric}={value!r}"
    return True, None


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")
        file.flush()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run_workload(
    args: argparse.Namespace,
    config: ServerConfig,
    stage: str,
    workload: Workload,
    run_dir: Path,
    server_process: subprocess.Popen[Any],
) -> TrialResult:
    workload_dir = run_dir / stage / workload.name
    workload_dir.mkdir(parents=True, exist_ok=True)
    raw_path = workload_dir / "benchmark.jsonl"
    log_path = workload_dir / "benchmark.log"
    if raw_path.exists():
        raw_path.unlink()

    command = build_benchmark_command(args, workload, raw_path)
    (workload_dir / "command.txt").write_text(shlex.join(command) + "\n")
    # An empty successful update resets the scheduler's cumulative speculative
    # counters.  The benchmark's accept_length then describes this workload
    # (including its warmup), instead of every workload run since server start.
    try:
        http_post_json(
            base_url(args) + "/set_internal_state",
            {"server_args": {}},
            timeout=10.0,
        )
    except (OSError, ValueError, urllib.error.URLError):
        # Older endpoints may not expose this control. Throughput/latency remain
        # valid; only accepted length will be cumulative for that server run.
        pass
    sampler = GpuSampler(args.nvidia_smi_gpu, args.telemetry_interval)
    start = time.monotonic()
    sampler.start()
    status = "failed"
    error: str | None = None
    metrics: dict[str, Any] = {}
    output_hash: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.benchmark_timeout,
                env=config_environment(os.environ, config, args.cuda_visible_devices),
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark exited with status {completed.returncode}\n{tail_text(log_path)}"
            )
        metrics = read_single_jsonl(raw_path)
        valid, validation_error = validate_benchmark_result(metrics, workload)
        if not valid:
            raise RuntimeError(validation_error or "benchmark validation failed")
        if server_process.poll() is not None:
            raise RuntimeError(
                f"server exited during benchmark with status {server_process.returncode}"
            )
        if not server_ready(base_url(args)):
            raise RuntimeError("server failed its health check after benchmark")
        output_hash = hash_outputs(metrics)
        status = "ok"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
    telemetry = sampler.stop()
    return TrialResult(
        config_id=config.config_id,
        config_label=config.label,
        stage=stage,
        workload=workload.name,
        workload_id=workload.workload_id,
        status=status,
        elapsed_s=time.monotonic() - start,
        config=asdict(config),
        workload_config=asdict(workload),
        metrics=metrics,
        telemetry=telemetry,
        output_hash=output_hash,
        error=error,
    )


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return results


def result_key(config_id: str, stage: str, workload_id: str) -> tuple[str, str, str]:
    return config_id, stage, workload_id


def completed_result_index(
    results: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        workload_id = result.get("workload_id")
        if workload_id is None and result.get("workload_config"):
            payload = json.dumps(
                result["workload_config"], sort_keys=True, separators=(",", ":")
            ).encode()
            workload_id = hashlib.sha256(payload).hexdigest()[:12]
        if workload_id is None:
            # Compatibility with early result files. Such a record will only
            # match another legacy lookup, never a new shape-aware workload.
            workload_id = result["workload"]
        key = result_key(result["config_id"], result["stage"], workload_id)
        index[key] = result
    return index


def check_log_for_fatal_errors(path: Path) -> str | None:
    content = tail_text(path, max_bytes=2_000_000).lower()
    for pattern in FATAL_LOG_PATTERNS:
        if pattern in content:
            return pattern
    return None


def should_run_trial(previous: dict[str, Any] | None, skip_failed: bool) -> bool:
    if previous is None:
        return True
    if previous.get("status") == "ok":
        return False
    return not skip_failed


def run_config(
    args: argparse.Namespace,
    config: ServerConfig,
    stage: str,
    workloads: Sequence[Workload],
    results_path: Path,
    result_index: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    pending: list[Workload] = []
    for workload in workloads:
        previous = result_index.get(
            result_key(config.config_id, stage, workload.workload_id)
        )
        if should_run_trial(previous, args.skip_failed):
            pending.append(workload)
    if not pending:
        print(f"[{config.label}/{stage}] already complete; skipping")
        return

    run_dir = args.output_dir / "runs" / f"{config.config_id}_{safe_name(config.label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_dir / f"server_{stage}.log"
    command = build_server_command(args, config)
    env = config_environment(os.environ, config, args.cuda_visible_devices)
    metadata = {
        "created_at": utc_now(),
        "stage": stage,
        "config": asdict(config),
        "command": command,
        "environment": {
            key: env.get(key)
            for key in sorted(
                set(DEBUG_ENV_VARS)
                | {
                    "CUDA_VISIBLE_DEVICES",
                    "KT_MXFP4_BACKEND",
                    "KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT",
                    "SGLANG_OPT_USE_TILELANG_MHC_POST",
                    "SGLANG_OPT_FUSE_MHC_POST_PRE",
                    "SGLANG_DSPARK_ENABLE_MULTI_STREAM",
                    "SGLANG_RAGGED_VERIFY_MODE",
                }
            )
        },
    }
    (run_dir / f"metadata_{stage}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"[{config.label}/{stage}] starting server for {len(pending)} workload(s): "
        f"{config.config_id}"
    )
    process: subprocess.Popen[Any] | None = None
    startup_s = 0.0
    startup_error: str | None = None
    workload_error: str | None = None
    try:
        with server_log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
        startup_s = wait_for_server(
            process,
            base_url(args),
            args.server_startup_timeout,
            server_log_path,
        )
        print(f"[{config.label}/{stage}] ready in {startup_s:.1f}s")
        for workload in pending:
            print(
                f"  {workload.name}: input={workload.input_len} "
                f"output={workload.output_len} concurrency={workload.concurrency}"
            )
            result = run_workload(args, config, stage, workload, run_dir, process)
            record = asdict(result) | {
                "recorded_at": utc_now(),
                "server_startup_s": startup_s,
            }
            append_jsonl(results_path, record)
            result_index[result_key(config.config_id, stage, workload.workload_id)] = (
                record
            )
            metric = result.metrics.get("output_throughput")
            suffix = (
                f" output_tps={metric:.2f}" if isinstance(metric, (int, float)) else ""
            )
            print(f"    -> {result.status}{suffix}")
            if result.status != "ok" or process.poll() is not None:
                workload_error = result.error or (
                    f"server exited with status {process.returncode}"
                )
                break
    except (OSError, RuntimeError, TimeoutError) as exc:
        startup_error = str(exc)
    finally:
        if process is not None:
            stop_process_group(process, args.server_shutdown_timeout)
        deadline = time.monotonic() + args.server_shutdown_timeout
        while port_is_open(args.host, args.port) and time.monotonic() < deadline:
            time.sleep(1.0)

    fatal_log_error = check_log_for_fatal_errors(server_log_path)
    if startup_error or workload_error or fatal_log_error:
        reason = (
            startup_error
            or workload_error
            or f"fatal server log pattern: {fatal_log_error}"
        )
        print(f"[{config.label}/{stage}] failed: {reason}")
        existing = completed_result_index(load_results(results_path))
        for workload in pending:
            key = result_key(config.config_id, stage, workload.workload_id)
            if key in existing:
                continue
            record = asdict(
                TrialResult(
                    config_id=config.config_id,
                    config_label=config.label,
                    stage=stage,
                    workload=workload.name,
                    workload_id=workload.workload_id,
                    status="failed",
                    elapsed_s=0.0,
                    config=asdict(config),
                    workload_config=asdict(workload),
                    metrics={},
                    telemetry={},
                    error=reason,
                )
            ) | {"recorded_at": utc_now(), "server_startup_s": startup_s}
            append_jsonl(results_path, record)
            result_index[key] = record


def positive_metric(result: dict[str, Any], name: str) -> float | None:
    try:
        value = float(result["metrics"][name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def scenario_ratio(
    result: dict[str, Any], baseline: dict[str, Any], latency_focus: bool
) -> float | None:
    output_tps = positive_metric(result, "output_throughput")
    baseline_output_tps = positive_metric(baseline, "output_throughput")
    tpot = positive_metric(result, "median_tpot_ms")
    baseline_tpot = positive_metric(baseline, "median_tpot_ms")
    p95 = positive_metric(result, "p95_e2e_latency_ms")
    baseline_p95 = positive_metric(baseline, "p95_e2e_latency_ms")
    ttft = positive_metric(result, "median_ttft_ms")
    baseline_ttft = positive_metric(baseline, "median_ttft_ms")
    values = (output_tps, baseline_output_tps, tpot, baseline_tpot, p95, baseline_p95)
    if any(value is None for value in values):
        return None

    if latency_focus and ttft is not None and baseline_ttft is not None:
        terms = (
            (output_tps / baseline_output_tps, 0.35),
            (baseline_ttft / ttft, 0.40),
            (baseline_p95 / p95, 0.25),
        )
    else:
        terms = (
            (output_tps / baseline_output_tps, 0.60),
            (baseline_tpot / tpot, 0.25),
            (baseline_p95 / p95, 0.15),
        )
    return math.exp(sum(weight * math.log(max(ratio, 1e-9)) for ratio, weight in terms))


def rank_configs(
    configs: Sequence[ServerConfig],
    workloads: Sequence[Workload],
    results: Sequence[dict[str, Any]],
    stage: str,
    require_output_match: bool,
) -> list[dict[str, Any]]:
    index = completed_result_index(results)
    baseline_config = next(
        (config for config in configs if config.label == "baseline"), configs[0]
    )
    baseline_results = {
        workload.name: index.get(
            result_key(baseline_config.config_id, stage, workload.workload_id)
        )
        for workload in workloads
    }
    ranking: list[dict[str, Any]] = []
    for config in configs:
        total_log_score = 0.0
        total_weight = 0.0
        failures: list[str] = []
        mismatches: list[str] = []
        accept_lengths: list[float] = []
        output_tps: dict[str, float] = {}
        for workload in workloads:
            result = index.get(
                result_key(config.config_id, stage, workload.workload_id)
            )
            baseline = baseline_results.get(workload.name)
            if (
                result is None
                or baseline is None
                or result.get("status") != "ok"
                or baseline.get("status") != "ok"
            ):
                failures.append(workload.name)
                continue
            ratio = scenario_ratio(result, baseline, workload.latency_focus)
            if ratio is None:
                failures.append(workload.name)
                continue
            total_log_score += workload.weight * math.log(max(ratio, 1e-9))
            total_weight += workload.weight
            throughput = positive_metric(result, "output_throughput")
            if throughput is not None:
                output_tps[workload.name] = throughput
            accept_length = positive_metric(result, "accept_length")
            if accept_length is not None:
                accept_lengths.append(accept_length)
            if (
                result.get("output_hash")
                and baseline.get("output_hash")
                and result["output_hash"] != baseline["output_hash"]
            ):
                mismatches.append(workload.name)

        complete = not failures and total_weight > 0
        if require_output_match and mismatches:
            complete = False
        score = 100.0 * math.exp(total_log_score / total_weight) if complete else 0.0
        ranking.append(
            {
                "config_id": config.config_id,
                "label": config.label,
                "score": score,
                "complete": complete,
                "failed_workloads": failures,
                "output_mismatches": mismatches,
                "mean_accept_length": (
                    statistics.fmean(accept_lengths) if accept_lengths else None
                ),
                "output_throughput": output_tps,
                "config": asdict(config),
            }
        )
    return sorted(
        ranking, key=lambda item: (item["complete"], item["score"]), reverse=True
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "config"


def write_ranking(path: Path, ranking: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "config_id",
                "label",
                "score",
                "complete",
                "mean_accept_length",
                "failed_workloads",
                "output_mismatches",
                "config_json",
            ]
        )
        for rank, item in enumerate(ranking, 1):
            writer.writerow(
                [
                    rank,
                    item["config_id"],
                    item["label"],
                    f"{item['score']:.6f}",
                    item["complete"],
                    item["mean_accept_length"],
                    ",".join(item["failed_workloads"]),
                    ",".join(item["output_mismatches"]),
                    json.dumps(item["config"], sort_keys=True),
                ]
            )


def select_finalists(
    configs: Sequence[ServerConfig],
    ranking: Sequence[dict[str, Any]],
    count: int,
) -> list[ServerConfig]:
    by_id = {config.config_id: config for config in configs}
    selected: list[ServerConfig] = []
    baseline = next((config for config in configs if config.label == "baseline"), None)
    if baseline is not None:
        selected.append(baseline)
    for item in ranking:
        config = by_id[item["config_id"]]
        if not item["complete"] or config in selected:
            continue
        selected.append(config)
        if len(selected) >= count:
            break
    return selected


def capture_command(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def write_system_info(args: argparse.Namespace) -> None:
    info = {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": args.python,
        "cpu_topology": detect_cpu_topology(),
        "gpu_numa_node": detect_gpu_numa_node(args.nvidia_smi_gpu),
        "commands": [
            capture_command(["nvidia-smi", "-L"]),
            capture_command(["nvidia-smi", "topo", "-m"]),
            capture_command(
                [
                    "nvidia-smi",
                    "-i",
                    args.nvidia_smi_gpu,
                    "--query-gpu=name,driver_version,compute_cap,memory.total,power.limit",
                    "--format=csv,noheader",
                ]
            ),
            capture_command(
                [
                    args.python,
                    "-c",
                    "import torch, flashinfer; "
                    "print(torch.__version__, torch.version.cuda, flashinfer.__version__)",
                ]
            ),
        ],
    }
    (args.output_dir / "system_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )


def write_best_config(
    args: argparse.Namespace,
    config: ServerConfig,
    ranking_item: dict[str, Any],
) -> None:
    command = build_server_command(args, config)
    env = config_environment({}, config, args.cuda_visible_devices)
    payload = {
        "selected_at": utc_now(),
        "ranking": ranking_item,
        "config": asdict(config),
        "environment": env,
        "command": command,
    }
    (args.output_dir / "best_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append("unset " + " ".join((*DEBUG_ENV_VARS, "SGL_REPO")))
    lines.append("")
    for name, value in sorted(env.items()):
        lines.append(f"export {name}={shlex.quote(value)}")
    lines.extend(["", f"exec {shlex.join(command)}", ""])
    launch_path = args.output_dir / "launch_best.sh"
    launch_path.write_text("\n".join(lines))
    launch_path.chmod(0o755)


def print_plan(
    candidates: Sequence[ServerConfig],
    search_workloads: Sequence[Workload],
    stress_workloads: Sequence[Workload],
) -> None:
    print(f"Server candidates ({len(candidates)}):")
    cpuinfer_values = unique_preserving_order(
        config.cpu_layout.cpuinfer_threads for config in candidates
    )
    print(
        "KT CPUInfer thread counts represented: "
        + ",".join(str(value) for value in cpuinfer_values)
    )
    for index, config in enumerate(candidates):
        print(
            f"  {index:02d} {config.config_id} {config.label}: "
            f"gpu_experts={config.gpu_experts} gamma={config.dspark_block_size} "
            f"amx_min={config.amx_min_tokens} cpu={config.cpu_layout.label}/"
            f"{config.cpu_layout.cpuinfer_threads}t numa={config.cpu_layout.numa_nodes} "
            f"prefill={config.gpu_prefill_threshold}/{config.mxfp4_prefill_slots}/"
            f"host{config.prefill_host_staging_experts} "
            f"chunk={config.chunked_prefill_size} max_run={config.max_running_requests} "
            f"mhc_fuse={config.fuse_mhc_post_pre} multistream={config.dspark_multistream}"
        )
    print("Search workloads:")
    for workload in search_workloads:
        print(f"  {asdict(workload)}")
    print("Finalist stress workloads:")
    for workload in stress_workloads:
        print(f"  {asdict(workload)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DSV4 DSpark + KTransformers native MXFP4 inference"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--kt-weight-path", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=Path("dspark_tuning"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--speculative-draft-device",
        help=(
            "Optional logical CUDA index or GPU UUID for the DSpark draft. "
            "When CUDA_VISIBLE_DEVICES=0,2, the second visible GPU is cuda:1; "
            "a UUID avoids remapping ambiguity."
        ),
    )
    parser.add_argument(
        "--nvidia-smi-gpu",
        default="0",
        help="Physical GPU index/UUID used only for telemetry and NUMA detection",
    )

    parser.add_argument(
        "--search-strategy", choices=("ofat", "cartesian"), default="ofat"
    )
    parser.add_argument("--max-configs", type=int, default=128)
    parser.add_argument(
        "--mixed-configs",
        type=int,
        default=8,
        help="Maximum mixed/interacting configurations added after the OFAT sweep",
    )
    parser.add_argument("--config-start", type=int, default=0)
    parser.add_argument("--config-end", type=int, default=None)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument(
        "--profile", choices=("smoke", "balanced", "exhaustive"), default="balanced"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Do not retry failed trials when resuming (successful trials are always skipped)",
    )
    parser.add_argument("--require-output-match", action="store_true")

    # The first value of every list defines the baseline.
    parser.add_argument("--gpu-experts", default="96,80,64")
    parser.add_argument("--amx-min-tokens", default="4,0,2,8")
    parser.add_argument("--gpu-prefill-thresholds", default="4096,2048,8192,0")
    parser.add_argument("--mxfp4-prefill-slots", default="auto,1")
    parser.add_argument("--prefill-host-staging-experts", default="8,16,4")
    parser.add_argument("--dspark-block-sizes", default="7,3,5")
    parser.add_argument("--chunked-prefill-sizes", default="4096,2048,8192")
    parser.add_argument("--max-running-requests", default="48,16,32")
    parser.add_argument("--mem-fractions", default="0.86")
    parser.add_argument("--placement-strategies", default="uniform,front-loading")
    parser.add_argument("--fuse-mhc-post-pre", default="false,true")
    parser.add_argument("--dspark-multistream", default="true,false")
    parser.add_argument("--ragged-verify-modes", default="static")
    parser.add_argument(
        "--cpuinfer-step",
        type=int,
        default=4,
        help="Thread increment for the automatic all-NUMA --kt-cpuinfer sweep",
    )
    parser.add_argument(
        "--cpuinfer-min",
        type=int,
        default=4,
        help="Minimum thread count for the automatic --kt-cpuinfer sweep",
    )
    parser.add_argument(
        "--cpuinfer-max",
        type=int,
        default=None,
        help="Maximum automatic --kt-cpuinfer threads; defaults to online logical CPUs",
    )
    parser.add_argument(
        "--cpu-layout",
        action="append",
        type=parse_custom_cpu_layout,
        help=(
            "Repeatable LABEL:THREADS:NODE,NODE; supplying one disables the "
            "automatic four-thread CPUInfer sweep"
        ),
    )

    parser.add_argument("--concurrencies", default="1,2,4,8,16,32")
    parser.add_argument("--large-contexts", default="8192,32768,131072,262144")
    parser.add_argument("--context-token-budget", type=int, default=300000)
    parser.add_argument("--decode-output-len", type=int, default=512)
    parser.add_argument("--context-output-len", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sps-table-path")
    parser.add_argument("--expert-frequency-path")
    parser.add_argument("--align-verify-to-graph-tier", action="store_true")
    parser.add_argument("--swa-full-tokens-ratio", type=float, default=0.1)
    parser.add_argument("--server-startup-timeout", type=float, default=3600)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument("--server-shutdown-timeout", type=float, default=45)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument(
        "extra_server_args",
        nargs=argparse.REMAINDER,
        help="Additional launch_server arguments; place them after --",
    )
    args = parser.parse_args(argv)
    if args.extra_server_args and args.extra_server_args[0] == "--":
        args.extra_server_args = args.extra_server_args[1:]
    if args.max_configs <= 0 or args.finalists <= 0:
        parser.error("--max-configs and --finalists must be positive")
    if args.mixed_configs < 0:
        parser.error("--mixed-configs must be non-negative")
    if args.cpuinfer_step <= 0 or args.cpuinfer_min <= 0:
        parser.error("--cpuinfer-step and --cpuinfer-min must be positive")
    if args.cpuinfer_max is not None and args.cpuinfer_max <= 0:
        parser.error("--cpuinfer-max must be positive")
    if args.cpuinfer_max is not None and args.cpuinfer_min > args.cpuinfer_max:
        parser.error("--cpuinfer-min must be <= --cpuinfer-max")
    if args.config_end is not None and args.config_end < args.config_start:
        parser.error("--config-end must be >= --config-start")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    candidates = build_candidates(args)
    if not candidates:
        raise RuntimeError("candidate slice is empty")
    search_workloads = build_search_workloads(args.profile)
    stress_workloads = build_stress_workloads(args)
    print_plan(candidates, search_workloads, stress_workloads)
    if args.dry_run:
        return 0

    if port_is_open(args.host, args.port):
        raise RuntimeError(
            f"{args.host}:{args.port} is already in use; choose a free --port. "
            "The tuner will not stop an existing server."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_system_info(args)
    (args.output_dir / "plan.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "arguments": {
                    key: json_safe(value) for key, value in vars(args).items()
                },
                "candidates": [asdict(config) for config in candidates],
                "search_workloads": [asdict(workload) for workload in search_workloads],
                "stress_workloads": [asdict(workload) for workload in stress_workloads],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    results_path = args.output_dir / "results.jsonl"
    if args.no_resume and results_path.exists():
        rotated = results_path.with_name(
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        results_path.rename(rotated)
    results = load_results(results_path)
    result_index = completed_result_index(results)

    for config in candidates:
        run_config(
            args,
            config,
            "search",
            search_workloads,
            results_path,
            result_index,
        )

    results = load_results(results_path)
    search_ranking = rank_configs(
        candidates,
        search_workloads,
        results,
        "search",
        args.require_output_match,
    )
    write_ranking(args.output_dir / "ranking_search.csv", search_ranking)
    finalists = select_finalists(candidates, search_ranking, args.finalists)
    if not finalists:
        raise RuntimeError("no configuration passed the search workload matrix")
    print("Finalists: " + ", ".join(config.label for config in finalists))

    for config in finalists:
        run_config(
            args,
            config,
            "stress",
            stress_workloads,
            results_path,
            result_index,
        )

    results = load_results(results_path)
    stress_ranking = rank_configs(
        finalists,
        stress_workloads,
        results,
        "stress",
        args.require_output_match,
    )
    write_ranking(args.output_dir / "ranking_stress.csv", stress_ranking)
    best_item = next((item for item in stress_ranking if item["complete"]), None)
    if best_item is None:
        raise RuntimeError("no finalist passed the complete stress workload matrix")
    best_config = next(
        config for config in finalists if config.config_id == best_item["config_id"]
    )
    write_best_config(args, best_config, best_item)
    print(
        f"Best configuration: {best_config.label} ({best_config.config_id}), "
        f"normalized score={best_item['score']:.2f}"
    )
    print(f"Launch script: {args.output_dir / 'launch_best.sh'}")
    mismatches = best_item["output_mismatches"]
    if mismatches:
        print(
            "WARNING: best configuration output hashes differed from baseline for: "
            + ", ".join(mismatches)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
