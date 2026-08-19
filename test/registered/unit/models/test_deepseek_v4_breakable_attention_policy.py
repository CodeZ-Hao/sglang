"""Unit tests for the DeepSeek-V4 breakable-attention routing policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4BreakableAttentionPolicy(CustomTestCase):
    def test_target_verify_without_prefill_context_stays_in_decode_graph(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with (
            patch.object(
                deepseek_v4, "is_in_breakable_cuda_graph", return_value=True
            ),
            patch.object(
                deepseek_v4,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
        ):
            self.assertFalse(
                deepseek_v4._should_use_breakable_attention(forward_batch)
            )

    def test_prefill_breakable_graph_keeps_eager_attention_seam(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with (
            patch.object(
                deepseek_v4, "is_in_breakable_cuda_graph", return_value=True
            ),
            patch.object(
                deepseek_v4,
                "get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(),
            ),
        ):
            self.assertTrue(
                deepseek_v4._should_use_breakable_attention(forward_batch)
            )


if __name__ == "__main__":
    unittest.main()
