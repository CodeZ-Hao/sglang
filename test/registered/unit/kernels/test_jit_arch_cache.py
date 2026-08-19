from unittest.mock import patch

from sglang.kernels.jit.utils.arch import (
    ArchInfo,
    get_jit_cuda_arch,
    override_jit_cuda_arch,
)
from sglang.kernels.jit.utils.common import cache_once


def test_jit_arch_follows_current_cuda_device():
    capabilities = {101: (12, 0), 102: (8, 9)}
    current_devices = iter((101, 102, 101))

    with (
        patch(
            "sglang.kernels.jit.utils.arch.torch.cuda.current_device",
            side_effect=lambda: next(current_devices),
        ),
        patch(
            "sglang.kernels.jit.utils.arch.torch.cuda.get_device_capability",
            side_effect=lambda device: capabilities[device],
        ) as get_capability,
        patch(
            "sglang.kernels.jit.utils.arch._cuda_arch_suffix",
            side_effect=lambda major, minor: "f" if (major, minor) == (12, 0) else "",
        ),
    ):
        assert get_jit_cuda_arch().target_name == "12.0f"
        assert get_jit_cuda_arch().target_name == "8.9"
        assert get_jit_cuda_arch().target_name == "12.0f"

    assert get_capability.call_count == 2


def test_jit_arch_override_is_nested_and_context_local():
    with (
        patch(
            "sglang.kernels.jit.utils.arch.torch.cuda.current_device", return_value=103
        ),
        patch(
            "sglang.kernels.jit.utils.arch.torch.cuda.get_device_capability",
            return_value=(8, 9),
        ),
    ):
        assert get_jit_cuda_arch().target_name == "8.9"
        with override_jit_cuda_arch(12, 0, "a"):
            assert get_jit_cuda_arch().target_name == "12.0a"
            with override_jit_cuda_arch(9, 0, "a"):
                assert get_jit_cuda_arch().target_name == "9.0a"
            assert get_jit_cuda_arch().target_name == "12.0a"
        assert get_jit_cuda_arch().target_name == "8.9"


def test_kernel_op_cache_is_partitioned_by_jit_arch():
    calls = []

    def module_loader():
        calls.append(len(calls))
        return object()

    # cache_once intentionally gives sglang.kernels.ops functions a per-arch
    # namespace while retaining process-wide behavior for ordinary utilities.
    module_loader.__module__ = "sglang.kernels.ops.test_fake"
    cached_loader = cache_once(module_loader)

    with patch(
        "sglang.kernels.jit.utils.arch.get_jit_cuda_arch",
        side_effect=(ArchInfo(12, 0, "f"), ArchInfo(8, 9, ""), ArchInfo(12, 0, "f")),
    ):
        sm120_module = cached_loader()
        sm89_module = cached_loader()
        assert cached_loader() is sm120_module

    assert sm89_module is not sm120_module
    assert len(calls) == 2
