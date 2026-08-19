from unittest.mock import patch

from sglang.srt.layers import layernorm


def test_arch_local_jit_rmsnorm_only_for_heterogeneous_secondary_gpu():
    capabilities = {0: (12, 0), 1: (8, 9), 2: (12, 0)}
    layernorm._use_arch_local_jit_rmsnorm.cache_clear()
    try:
        with (
            patch.object(layernorm, "_is_cuda", True),
            patch.object(
                layernorm.torch.cuda,
                "get_device_capability",
                side_effect=lambda device: capabilities[device],
            ),
        ):
            assert not layernorm._use_arch_local_jit_rmsnorm(0)
            assert layernorm._use_arch_local_jit_rmsnorm(1)
            assert not layernorm._use_arch_local_jit_rmsnorm(2)
    finally:
        layernorm._use_arch_local_jit_rmsnorm.cache_clear()


def test_arch_local_jit_rmsnorm_is_conservative_when_capability_query_fails():
    layernorm._use_arch_local_jit_rmsnorm.cache_clear()
    try:
        with (
            patch.object(layernorm, "_is_cuda", True),
            patch.object(
                layernorm.torch.cuda,
                "get_device_capability",
                side_effect=RuntimeError("device unavailable"),
            ),
        ):
            assert layernorm._use_arch_local_jit_rmsnorm(1)
    finally:
        layernorm._use_arch_local_jit_rmsnorm.cache_clear()
