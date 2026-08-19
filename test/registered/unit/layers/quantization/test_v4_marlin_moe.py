from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestV4MarlinMoe(unittest.TestCase):
    def test_direct_repack_matches_gptq_transpose_bitwise(self):
        from sglang.kernels.ops.quantization.gptq_marlin_repack import (
            gptq_marlin_repack,
            mxfp4_marlin_repack,
        )

        for k, n in ((128, 128), (256, 192)):
            raw = torch.randint(
                0, 256, (3, n, k // 2), dtype=torch.uint8, device="cuda"
            )
            direct = mxfp4_marlin_repack(raw, k, n)
            perm = torch.empty(0, dtype=torch.int32, device="cuda")
            reference = torch.stack(
                [
                    gptq_marlin_repack(
                        raw[expert].view(torch.int32).view(n, k // 8).T.contiguous(),
                        perm,
                        k,
                        n,
                        4,
                    )
                    for expert in range(raw.shape[0])
                ]
            )
            torch.testing.assert_close(direct, reference, rtol=0, atol=0)

    def test_scale_swizzle_matches_marlin_reference_bitwise(self):
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_permute_scales,
        )
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            _swizzle_e8m0_scales,
        )

        e, k, n = 3, 256, 192
        exponents = torch.randint(-8, 8, (e, n, k // 32), device="cuda")
        scales = torch.exp2(exponents.float()).to(torch.float8_e8m0fnu)
        actual = _swizzle_e8m0_scales(scales, size_k=k, size_n=n)
        expected = []
        for expert in range(e):
            value = scales[expert].float().T.contiguous()
            value = marlin_permute_scales(value, k, n, 32)
            value = value.view(-1, 4)[:, [0, 2, 1, 3]].reshape(k // 32, n)
            expected.append(value.to(torch.float8_e8m0fnu))
        reference = torch.stack(expected)
        torch.testing.assert_close(
            actual.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
        )

    def test_moe_matches_reference_is_deterministic_and_graph_safe(self):
        from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            apply_v4_marlin_moe,
            prepare_v4_mxfp4_marlin,
        )

        torch.manual_seed(7)
        e, m, k, n, topk = 4, 5, 128, 128, 2
        w13 = torch.randint(
            0, 256, (e, 2 * n, k // 2), dtype=torch.uint8, device="cuda"
        )
        w2 = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device="cuda")
        s13 = torch.full((e, 2 * n, k // 32), 127, dtype=torch.uint8, device="cuda")
        s2 = torch.full((e, k, n // 32), 127, dtype=torch.uint8, device="cuda")
        prepared = prepare_v4_mxfp4_marlin(w13, s13, w2, s2)
        pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                prepared.w13,
                prepared.w13_scale,
                prepared.w2,
                prepared.w2_scale,
            )
        )
        prepare_v4_mxfp4_marlin(w13, s13, w2, s2, out=prepared)
        self.assertEqual(
            pointers,
            tuple(
                tensor.data_ptr()
                for tensor in (
                    prepared.w13,
                    prepared.w13_scale,
                    prepared.w2,
                    prepared.w2_scale,
                )
            ),
        )

        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.01
        ids = torch.tensor(
            [[0, 1], [2, 3], [1, -1], [3, 0], [2, 1]],
            dtype=torch.int32,
            device="cuda",
        )
        gates = torch.rand((m, topk), dtype=torch.float32, device="cuda")
        gates[ids < 0] = 0
        dw13 = MXFP4QuantizeUtil.dequantize(w13, torch.bfloat16, s13, [32])
        dw2 = MXFP4QuantizeUtil.dequantize(w2, torch.bfloat16, s2, [32])

        for swiglu_limit in (None, 2.0):
            reference_fp32 = torch.zeros_like(x, dtype=torch.float32)
            for token in range(m):
                for route in range(topk):
                    expert = int(ids[token, route])
                    if expert < 0:
                        continue
                    first = (
                        (x[token].float() @ dw13[expert].float().T)
                        .to(torch.bfloat16)
                        .float()
                    )
                    gate, up = first[:n], first[n:]
                    if swiglu_limit is not None:
                        gate = gate.clamp(max=swiglu_limit)
                        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
                    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
                    expert_output = (
                        (activated.float() @ dw2[expert].float().T)
                        * gates[token, route]
                    ).to(torch.bfloat16)
                    reference_fp32[token] += expert_output.float()

            kwargs = dict(
                hidden_states=x,
                prepared=prepared,
                topk_weights=gates,
                topk_ids=ids,
                routed_scaling_factor=0.75,
                swiglu_limit=swiglu_limit,
            )
            reference = (reference_fp32 * 0.75).to(torch.bfloat16)
            actual = apply_v4_marlin_moe(**kwargs).clone()
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.05)
            cosine = torch.nn.functional.cosine_similarity(
                actual.float().flatten(), reference.float().flatten(), dim=0
            )
            self.assertGreater(float(cosine), 0.999)
            for _ in range(5):
                repeated = apply_v4_marlin_moe(**kwargs).clone()
                torch.testing.assert_close(repeated, actual, rtol=0, atol=0)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = apply_v4_marlin_moe(**kwargs)
            graph.replay()
            torch.testing.assert_close(captured, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
