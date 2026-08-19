import functools
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
import sglang.srt.speculative.dspark_components.dspark_worker_v2 as dspark_worker_module
from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    fp8_gemm_runner_backend_context,
    get_fp8_gemm_runner_backend,
)
from sglang.srt.speculative.draft_worker_common import (
    _FLASHINFER_MULTIARCH_SAMPLING_READY,
    ensure_flashinfer_sampling_multiarch,
    resolve_speculative_draft_device,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResolveSpeculativeDraftDevice(CustomTestCase):
    def test_logical_indices(self):
        with patch("torch.cuda.device_count", return_value=3):
            self.assertEqual(resolve_speculative_draft_device("2"), 2)
            self.assertEqual(resolve_speculative_draft_device("cuda:1"), 1)
            with self.assertRaisesRegex(ValueError, "only 3 CUDA devices"):
                resolve_speculative_draft_device("3")

    def test_gpu_uuid(self):
        uuids = [
            "GPU-56fd682e-006d-2a13-a642-31748746dba6",
            "GPU-b6bb9e3a-d439-ef36-dcf5-40eeb5870765",
        ]
        with (
            patch("torch.cuda.device_count", return_value=2),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=lambda index: SimpleNamespace(uuid=uuids[index]),
            ),
        ):
            self.assertEqual(resolve_speculative_draft_device(uuids[1]), 1)

    def test_fp8_backend_override_is_scoped_to_draft_context(self):
        with patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.CUTLASS,
        ):
            self.assertEqual(
                get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.CUTLASS
            )
            with fp8_gemm_runner_backend_context("triton"):
                self.assertEqual(
                    get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.TRITON
                )
            self.assertEqual(
                get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.CUTLASS
            )

    def test_draft_context_disables_target_expert_distribution_recorder(self):
        class FakeRecorder:
            disabled_depth = 0

            @contextmanager
            def disable_this_region(self):
                self.disabled_depth += 1
                try:
                    yield
                finally:
                    self.disabled_depth -= 1

        recorder = FakeRecorder()
        worker = object.__new__(DSparkWorkerV2)
        worker._draft_dp_context_enabled = False
        worker.draft_gpu_id = 1
        worker._draft_fp8_gemm_backend = "auto"

        with (
            patch.object(
                dspark_worker_module,
                "draft_cuda_device_context",
                side_effect=lambda *_args: nullcontext(),
            ),
            patch.object(
                dspark_worker_module,
                "speculative_moe_backend_context",
                side_effect=nullcontext,
            ),
            patch.object(
                dspark_worker_module,
                "fp8_gemm_runner_backend_context",
                side_effect=lambda *_args: nullcontext(),
            ),
            patch.object(
                dspark_worker_module,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            with worker._draft_context():
                self.assertEqual(recorder.disabled_depth, 1)

        self.assertEqual(recorder.disabled_depth, 0)

    def test_disabled_recorder_skips_nested_forward_lifecycle(self):
        recorder = object.__new__(_ExpertDistributionRecorderReal)
        recorder._disable_all = False
        events = []
        recorder._on_forward_pass_start = lambda _batch: events.append("start")
        recorder._on_forward_pass_end = lambda _pass_id, _outputs: events.append("end")

        with recorder.disable_this_region():
            with recorder.with_forward_pass(1, SimpleNamespace()) as outputs:
                self.assertEqual(outputs, {})

        self.assertEqual(events, [])

    def test_flashinfer_sampling_is_forced_to_multiarch_jit(self):
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.__path__ = []
        fake_flashinfer.__version__ = "0.6.15.post1"

        fake_context_module = ModuleType("flashinfer.compilation_context")

        class FakeCompilationContext:
            @staticmethod
            def _normalize_cuda_arch(major, minor):
                return (major, "0f" if (major, minor) == (12, 0) else str(minor))

        fake_context_module.CompilationContext = FakeCompilationContext

        calls = []
        fake_sampling = ModuleType("flashinfer.sampling")
        fake_jit = ModuleType("flashinfer.jit")
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_env = SimpleNamespace(
                FLASHINFER_CACHE_DIR=Path(temp_dir),
                FLASHINFER_AOT_DIR=Path(temp_dir) / "original-aot",
                FLASHINFER_JIT_DIR=Path(temp_dir) / "original-jit",
            )

            @functools.cache
            def fake_get_sampling_module():
                calls.append(
                    (
                        os.environ.get("FLASHINFER_CUDA_ARCH_LIST"),
                        fake_env.FLASHINFER_AOT_DIR,
                        fake_env.FLASHINFER_JIT_DIR,
                    )
                )
                return object()

            fake_sampling.get_sampling_module = fake_get_sampling_module
            fake_jit.env = fake_env
            fake_flashinfer.sampling = fake_sampling
            fake_flashinfer.jit = fake_jit

            modules = {
                "flashinfer": fake_flashinfer,
                "flashinfer.compilation_context": fake_context_module,
                "flashinfer.jit": fake_jit,
                "flashinfer.sampling": fake_sampling,
            }
            original_aot = fake_env.FLASHINFER_AOT_DIR
            original_jit = fake_env.FLASHINFER_JIT_DIR
            _FLASHINFER_MULTIARCH_SAMPLING_READY.clear()
            try:
                with (
                    patch.dict(sys.modules, modules),
                    patch.dict(
                        os.environ,
                        {"FLASHINFER_CUDA_ARCH_LIST": "12.0f"},
                    ),
                    patch(
                        "torch.cuda.get_device_capability",
                        side_effect=lambda device: {0: (12, 0), 1: (8, 9)}[device],
                    ),
                ):
                    ensure_flashinfer_sampling_multiarch((0, 1))
                    ensure_flashinfer_sampling_multiarch((1, 0))
                    self.assertEqual(os.environ["FLASHINFER_CUDA_ARCH_LIST"], "12.0f")
            finally:
                _FLASHINFER_MULTIARCH_SAMPLING_READY.clear()

            self.assertEqual(len(calls), 1)
            arch_list, forced_aot, forced_jit = calls[0]
            self.assertEqual(arch_list, "8.9 12.0f")
            self.assertEqual(forced_aot.name, "no_aot")
            self.assertEqual(forced_jit.name, "cached_ops")
            self.assertEqual(fake_env.FLASHINFER_AOT_DIR, original_aot)
            self.assertEqual(fake_env.FLASHINFER_JIT_DIR, original_jit)


if __name__ == "__main__":
    unittest.main()
