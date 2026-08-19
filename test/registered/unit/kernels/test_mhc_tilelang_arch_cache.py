from unittest.mock import patch

from sglang.kernels.jit.utils.arch import ArchInfo
from sglang.kernels.ops.layernorm import mhc


class _FakeTilelang:
    def __init__(self):
        self.compile_kwargs = []

    def jit(self, **jit_kwargs):
        self.compile_kwargs.append(jit_kwargs)

        def decorate(_fn):
            target = jit_kwargs.get("target", "auto")
            return lambda: target

        return decorate


def test_current_tilelang_cuda_arch_follows_jit_device():
    with (
        patch.object(mhc.torch.version, "cuda", "13.0"),
        patch.object(mhc, "get_jit_cuda_arch", return_value=ArchInfo(12, 0, "f")),
    ):
        assert mhc._current_tilelang_cuda_arch() == "sm_120a"

    with (
        patch.object(mhc.torch.version, "cuda", "13.0"),
        patch.object(mhc, "get_jit_cuda_arch", return_value=ArchInfo(8, 9, "")),
    ):
        assert mhc._current_tilelang_cuda_arch() == "sm_89"


def test_lazy_tilelang_compiles_once_per_cuda_arch():
    fake_tilelang = _FakeTilelang()
    lazy_tilelang = mhc._LazyTilelang()

    @lazy_tilelang.jit(pass_configs={"example": True})
    def fake_kernel():
        raise AssertionError("TileLang should replace this function")

    with (
        patch.object(mhc, "_load_tilelang", return_value=fake_tilelang),
        patch.object(
            mhc,
            "_current_tilelang_cuda_arch",
            side_effect=("sm_120a", "sm_89", "sm_120a"),
        ),
    ):
        assert fake_kernel()["arch"] == "sm_120a"
        assert fake_kernel()["arch"] == "sm_89"
        assert fake_kernel()["arch"] == "sm_120a"

    assert fake_tilelang.compile_kwargs == [
        {
            "pass_configs": {"example": True},
            "target": {"kind": "cuda", "arch": "sm_120a"},
        },
        {
            "pass_configs": {"example": True},
            "target": {"kind": "cuda", "arch": "sm_89"},
        },
    ]


def test_mhc_split_heuristic_uses_current_device():
    properties = type("Properties", (), {"multi_processor_count": 128})()
    with (
        patch.object(mhc.torch.cuda, "current_device", return_value=2),
        patch.object(
            mhc.torch.cuda, "get_device_properties", return_value=properties
        ) as get_properties,
    ):
        mhc._compute_num_split_for_mhc_pre(64, 4096)

    get_properties.assert_called_once_with(2)
