import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
TUNER_PATH = REPO_ROOT / "scripts" / "tune_dsv4_dspark_kt.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves forward annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "ci_register_dspark_tuner", CI_REGISTER_PATH
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkKTTuner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = _load_module("tune_dsv4_dspark_kt", TUNER_PATH)

    def make_args(self, *extra):
        return self.tuner.parse_args(
            [
                "--model-path",
                "model",
                "--kt-weight-path",
                "weights",
                "--cpu-layout",
                "all:64:0,1",
                *extra,
            ]
        )

    def test_bounded_search_covers_each_knob(self):
        args = self.make_args("--max-configs", "24", "--mixed-configs", "9")
        candidates = self.tuner.build_candidates(args)

        self.assertEqual(len(candidates), 24)
        self.assertEqual(candidates[0].label, "baseline")
        self.assertEqual(len({candidate.config_id for candidate in candidates}), 24)
        labels = {candidate.label.split("=", 1)[0] for candidate in candidates}
        self.assertTrue(
            {
                "gpu_experts",
                "amx_min_tokens",
                "gpu_prefill_threshold",
                "mxfp4_prefill_slots",
                "prefill_host_staging_experts",
                "dspark_block_size",
                "chunked_prefill_size",
                "max_running_requests",
                "placement",
                "fuse_mhc_post_pre",
                "dspark_multistream",
            }.issubset(labels)
        )

    def test_auto_cpuinfer_sweep_uses_four_thread_intervals(self):
        topology = {
            "nodes": {0: list(range(32)), 1: list(range(32, 64))},
            "logical_cpus": 64,
            "physical_cores": 32,
            "physical_cores_by_node": {0: 16, 1: 16},
        }
        with (
            patch.object(self.tuner, "detect_cpu_topology", return_value=topology),
            patch.object(self.tuner, "detect_gpu_numa_node", return_value=0),
        ):
            layouts = self.tuner.auto_cpu_layouts(
                "0", cpuinfer_step=4, cpuinfer_min=4, cpuinfer_max=32
            )

        all_numa_threads = [
            layout.cpuinfer_threads for layout in layouts if layout.numa_nodes == (0, 1)
        ]
        self.assertEqual(all_numa_threads, [64, 4, 8, 12, 16, 20, 24, 28, 32])

    def test_long_context_matrix_respects_token_budget(self):
        args = self.make_args("--profile", "exhaustive")
        workloads = self.tuner.build_stress_workloads(args)

        self.assertTrue(any(item.concurrency == 32 for item in workloads))
        self.assertTrue(any(item.input_len == 262144 for item in workloads))
        self.assertTrue(
            all(
                item.input_len * item.concurrency <= args.context_token_budget
                for item in workloads
                if item.input_len > 128
            )
        )

    def test_launch_uses_native_mxfp4_and_strips_debug_environment(self):
        args = self.make_args("--max-configs", "1")
        config = self.tuner.build_candidates(args)[0]
        command = self.tuner.build_server_command(args, config)
        environment = self.tuner.config_environment(
            {
                "SGLANG_BCG_DEBUG_REPLAY": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
                "SGL_REPO": "/tmp/sglang",
            },
            config,
            "0",
        )

        self.assertIn("MXFP4", command)
        self.assertNotIn("AMXINT4", command)
        self.assertIn("breakable", command)
        self.assertIn("--kt-gpu-prefill-token-threshold", command)
        self.assertIn("--kt-mxfp4-prefill-slots", command)
        self.assertIn("--kt-mxfp4-prefill-host-staging-experts", command)
        self.assertNotIn("SGLANG_BCG_DEBUG_REPLAY", environment)
        self.assertNotIn("CUDA_LAUNCH_BLOCKING", environment)
        self.assertNotIn("SGL_REPO", environment)
        self.assertEqual(environment["SGLANG_REPO"], "/tmp/sglang")
        self.assertEqual(environment["KT_MXFP4_BACKEND"], "amx")

    def test_benchmark_uses_exact_requested_token_lengths(self):
        args = self.make_args()
        workload = self.tuner.Workload("fixed", 128, 256, 4, 8)
        command = self.tuner.build_benchmark_command(
            args, workload, Path("benchmark.jsonl")
        )

        ratio_index = command.index("--random-range-ratio") + 1
        self.assertEqual(command[ratio_index], "1")

    def test_optional_remote_draft_uses_marlin(self):
        args = self.make_args(
            "--cuda-visible-devices",
            "0,2",
            "--speculative-draft-device",
            "GPU-b6bb9e3a-d439-ef36-dcf5-40eeb5870765",
        )
        config = self.tuner.build_candidates(args)[0]
        command = self.tuner.build_server_command(args, config)

        self.assertIn("--speculative-draft-device", command)
        self.assertIn("GPU-b6bb9e3a-d439-ef36-dcf5-40eeb5870765", command)
        backend_index = command.index("--speculative-moe-runner-backend") + 1
        self.assertEqual(command[backend_index], "marlin")

    def test_frequency_profile_does_not_enable_expert_remapping(self):
        args = self.make_args(
            "--placement-strategies",
            "frequency",
            "--expert-frequency-path",
            "/profiles/recording.pt",
        )
        config = self.tuner.build_candidates(args)[0]
        command = self.tuner.build_server_command(args, config)

        profile_index = command.index("--kt-expert-frequency-file") + 1
        location_index = command.index("--init-expert-location") + 1
        self.assertEqual(command[profile_index], "/profiles/recording.pt")
        self.assertEqual(command[location_index], "trivial")

    def test_arguments_are_json_serializable(self):
        args = self.make_args()
        json.dumps(self.tuner.json_safe(vars(args)))

    def test_resume_identity_includes_full_workload_shape(self):
        smoke = self.tuner.build_search_workloads("smoke")[0]
        balanced = self.tuner.build_search_workloads("balanced")[0]

        self.assertEqual(smoke.name, balanced.name)
        self.assertNotEqual(smoke.workload_id, balanced.workload_id)

    def test_resume_retries_failures_but_not_successes(self):
        self.assertTrue(self.tuner.should_run_trial(None, skip_failed=False))
        self.assertTrue(
            self.tuner.should_run_trial({"status": "failed"}, skip_failed=False)
        )
        self.assertFalse(
            self.tuner.should_run_trial({"status": "failed"}, skip_failed=True)
        )
        self.assertFalse(
            self.tuner.should_run_trial({"status": "ok"}, skip_failed=False)
        )

    def test_ranking_rewards_throughput_and_latency(self):
        args = self.make_args("--max-configs", "2")
        configs = self.tuner.build_candidates(args)
        workload = self.tuner.Workload("decode", 128, 128, 1, 2)

        def record(config, throughput, tpot, p95):
            return {
                "config_id": config.config_id,
                "stage": "search",
                "workload": workload.name,
                "workload_id": workload.workload_id,
                "status": "ok",
                "metrics": {
                    "output_throughput": throughput,
                    "median_tpot_ms": tpot,
                    "p95_e2e_latency_ms": p95,
                    "median_ttft_ms": 10,
                },
                "output_hash": "same",
            }

        ranking = self.tuner.rank_configs(
            configs,
            [workload],
            [record(configs[0], 10, 100, 1000), record(configs[1], 12, 80, 800)],
            "search",
            False,
        )

        self.assertEqual(ranking[0]["config_id"], configs[1].config_id)
        self.assertGreater(ranking[0]["score"], 100)


if __name__ == "__main__":
    unittest.main()
