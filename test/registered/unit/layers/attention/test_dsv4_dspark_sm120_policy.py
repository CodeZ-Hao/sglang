"""Unit tests for the DSV4 DSpark SM120 sparse-attention width policy."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.deepseek_v4_backend as dsv4_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDsv4DsparkSm120Policy(CustomTestCase):
    def test_flashinfer_uses_an_instantiated_dsv4_topk(self):
        with (
            patch.object(dsv4_backend, "_is_cuda", True),
            patch.object(
                dsv4_backend.torch.cuda,
                "get_device_capability",
                return_value=(12, 0),
            ),
            patch.dict(
                os.environ,
                {"SGLANG_SM120_FLASHMLA_BACKEND": "flashinfer"},
            ),
        ):
            # A six-token DSpark block needs 134 logical entries, naturally
            # aligned to 192. FlashInfer has no 192-wide DSV4 specialization.
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=6),
                512,
            )
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=384),
                512,
            )
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=385),
                1024,
            )
            with self.assertRaisesRegex(ValueError, "exceeds.*envelope"):
                dsv4_backend._dspark_swa_page_index_alignment(block_size=897)

    def test_other_backends_keep_compact_64_alignment(self):
        with (
            patch.object(dsv4_backend, "_is_cuda", True),
            patch.object(
                dsv4_backend.torch.cuda,
                "get_device_capability",
                return_value=(12, 0),
            ),
            patch.dict(
                os.environ,
                {"SGLANG_SM120_FLASHMLA_BACKEND": "triton"},
            ),
        ):
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=6),
                64,
            )

    def test_sm8x_fallback_masks_padding_and_applies_sink(self):
        backend = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(head_dim=4)),
            softmax_scale=0.5,
        )
        q = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.bfloat16)
        table = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [100, 100, 100, 100]],
            dtype=torch.bfloat16,
        )
        indices = torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)

        def fake_dequant(_cache, flat_indices, page_size):
            del page_size
            return table[flat_indices.long()].view(-1, 1, 4)

        with patch.object(
            dsv4_backend, "dequantize_k_cache_paged", side_effect=fake_dequant
        ):
            actual = dsv4_backend.DeepseekV4AttnBackend._forward_dspark_sm8x_swa(
                backend,
                q=q,
                quant_k_cache=torch.empty(0),
                indices=indices,
                topk_length=lengths,
                page_size=128,
                attn_sink=torch.zeros(2),
            )

        scores = torch.matmul(q, table[:2].T).float() * 0.5
        probs = torch.softmax(
            torch.cat((scores, torch.zeros(1, 2, 1)), dim=-1), dim=-1
        )[..., :2]
        expected = torch.matmul(probs.to(torch.bfloat16), table[:2])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
