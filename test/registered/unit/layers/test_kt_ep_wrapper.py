from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTGraphStateBridge,
    _Mxfp4LayerwisePrefillManager,
    _moe_layer_indices,
    create_kt_config_from_server_args,
    mask_and_remap_expert_ids,
    partition_and_remap_expert_ids,
)


def test_graph_state_bridge_reuses_largest_stable_buffer():
    bridge = KTGraphStateBridge()
    large = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    large_view = bridge.copy("residual", large)
    stable_ptr = large_view.data_ptr()
    assert stable_ptr != large.data_ptr()
    torch.testing.assert_close(large_view, large)

    small = torch.full((2, 3), 7.0)
    small_view = bridge.copy("residual", small)
    assert small_view.data_ptr() == stable_ptr
    assert small_view.shape == small.shape
    torch.testing.assert_close(small_view, small)


def test_graph_state_bridge_keeps_replaced_buffers_alive():
    bridge = KTGraphStateBridge()
    old_view = bridge.copy("residual", torch.zeros((2, 3)))
    old_ptr = old_view.data_ptr()

    grown_view = bridge.copy("residual", torch.ones((4, 3)))

    assert grown_view.data_ptr() != old_ptr
    assert bridge._retired_buffers[0].data_ptr() == old_ptr
    torch.testing.assert_close(grown_view, torch.ones((4, 3)))


def test_mask_and_remap_arbitrary_gpu_experts_and_padding():
    gpu_mask = torch.tensor([False, False, True, False, False, True])
    logical_to_gpu = torch.tensor([-1, -1, 0, -1, -1, 1])
    logical_ids = torch.tensor([[2, 5, 1, -1, 6]])

    actual = mask_and_remap_expert_ids(logical_ids, gpu_mask, logical_to_gpu)

    torch.testing.assert_close(actual, torch.tensor([[0, 1, -1, -1, -1]]))


def test_mask_and_remap_compact_cpu_complement():
    cpu_mask = torch.tensor([True, True, False, True, True, False])
    logical_to_cpu = torch.tensor([0, 1, -1, 2, 3, -1])
    logical_ids = torch.tensor([[0, 2, 5, 4, -1, 6]])

    actual = mask_and_remap_expert_ids(logical_ids, cpu_mask, logical_to_cpu)

    torch.testing.assert_close(actual, torch.tensor([[0, -1, -1, 3, -1, -1]]))


def test_partition_and_remap_produces_distinct_complementary_outputs():
    gpu_mask = torch.tensor([False, False, True, False, False, True])
    logical_to_gpu = torch.tensor([-1, -1, 0, -1, -1, 1])
    logical_to_cpu = torch.tensor([0, 1, -1, 2, 3, -1])
    logical_ids = torch.tensor([[0, 2, 5, 4, -1, 6]])

    cpu_ids, gpu_ids = partition_and_remap_expert_ids(
        logical_ids, gpu_mask, logical_to_gpu, logical_to_cpu
    )

    torch.testing.assert_close(cpu_ids, torch.tensor([[0, -1, -1, 3, -1, -1]]))
    torch.testing.assert_close(gpu_ids, torch.tensor([[-1, 0, 1, -1, -1, -1]]))
    assert cpu_ids.data_ptr() != gpu_ids.data_ptr()


def test_layerwise_prefill_reorders_flashinfer_up_gate_to_marlin_gate_up():
    class Event:
        def record(self, stream):
            del stream

    manager = object.__new__(_Mxfp4LayerwisePrefillManager)
    manager.intermediate_size = 2
    manager.raw_staging = {
        "w13_weight": torch.empty((2, 4, 2), dtype=torch.int8),
        "w13_weight_scale_inv": torch.empty((2, 4, 1), dtype=torch.uint8),
        "w2_weight": torch.empty((2, 4, 1), dtype=torch.int8),
        "w2_weight_scale_inv": torch.empty((2, 4, 1), dtype=torch.uint8),
    }
    manager.host_is_pinned = False
    manager.transfer_stream = None
    manager.gpu_scale_free_events = [Event(), Event()]
    manager.gpu_scale_slot_used = [False, False]
    manager.gpu_scale_staging = {
        "w13_weight_scale_inv": torch.tensor(
            [[[9], [10], [11], [12]], [[0], [0], [0], [0]]]
        ),
        "w2_weight_scale_inv": torch.tensor(
            [[[17], [18], [19], [20]], [[0], [0], [0], [0]]]
        ),
    }
    method = SimpleNamespace(logical_to_gpu_index=torch.tensor([0]))
    layer = SimpleNamespace(
        w13_weight=torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]]),
        w2_weight=torch.tensor([[[13], [14], [15], [16]]]),
    )

    manager._stage_gpu_expert(method, layer, logical_id=0, raw_slot=0)

    assert manager.raw_staging["w13_weight"][0].tolist() == [
        [5, 6],
        [7, 8],
        [1, 2],
        [3, 4],
    ]
    assert manager.raw_staging["w13_weight_scale_inv"][0].tolist() == [
        [11],
        [12],
        [9],
        [10],
    ]


def test_layerwise_prefill_upload_uses_compact_cpu_expert_id():
    calls = []

    class Wrapper:
        def submit_write_weight_scale_to_buffer(self, *args):
            calls.append(args)

        def sync_write_weight_scale_to_buffer(self):
            calls.append("sync")

    manager = object.__new__(_Mxfp4LayerwisePrefillManager)
    manager.host_staging = {
        name: torch.empty((2, 1), dtype=torch.uint8)
        for name in _Mxfp4LayerwisePrefillManager._RAW_NAMES
    }
    method = SimpleNamespace(
        wrapper=Wrapper(), logical_to_cpu_index=torch.tensor([-1, 0, -1, 1])
    )

    manager._submit_cpu_expert(method, logical_id=3, host_slot=1)

    assert calls[0][0:2] == (1, 1)
    assert calls[1] == "sync"


def test_dsv4_hash_prefix_is_included_in_moe_layers():
    config = SimpleNamespace(
        num_hidden_layers=43,
        num_hash_layers=3,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )

    assert _moe_layer_indices(config) == list(range(43))


def test_sparse_moe_frequency_is_respected():
    config = SimpleNamespace(
        num_hidden_layers=8,
        num_hash_layers=0,
        first_k_dense_replace=2,
        moe_layer_freq=2,
    )

    assert _moe_layer_indices(config) == [2, 4, 6]


def test_frequency_placement_selects_top_experts_per_layer(tmp_path, monkeypatch):
    scores = torch.tensor(
        [
            [1, 9, 3, 8],
            [6, 2, 7, 0],
            [4, 5, 1, 10],
        ]
    )
    profile = tmp_path / "expert_distribution_recorder.pt"
    torch.save({"logical_count": scores}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=4,
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=2,
        kt_expert_placement_strategy="frequency",
        kt_expert_frequency_file=str(profile),
        init_expert_location="trivial",
    )
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0),
    )
    monkeypatch.setattr(kt_ep_wrapper.dist, "is_initialized", lambda: False)

    masks = kt_ep_wrapper._build_gpu_expert_masks(server_args)

    assert masks.tolist() == [
        [False, True, False, True],
        [True, False, True, False],
        [False, True, False, True],
    ]
    assert masks.sum(dim=1).tolist() == [2, 2, 2]


def test_frequency_placement_sums_recorder_buffer(tmp_path, monkeypatch):
    samples = torch.tensor(
        [
            [[10, 0, 0], [0, 2, 1]],
            [[0, 9, 0], [4, 0, 0]],
        ]
    )
    profile = tmp_path / "buffered_distribution.pt"
    torch.save({"logical_count": samples}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=3,
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=1,
        kt_expert_placement_strategy="frequency",
        kt_expert_frequency_file=str(profile),
        init_expert_location="trivial",
    )
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0),
    )
    monkeypatch.setattr(kt_ep_wrapper.dist, "is_initialized", lambda: False)

    masks = kt_ep_wrapper._build_gpu_expert_masks(server_args)

    assert masks.tolist() == [[True, False, False], [True, False, False]]


def test_frequency_placement_rejects_incomplete_profile(tmp_path, monkeypatch):
    profile = tmp_path / "incomplete_distribution.pt"
    torch.save({"logical_count": torch.tensor([[1, 2], [0, 0]])}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=2,
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=1,
        kt_expert_placement_strategy="frequency",
        kt_expert_frequency_file=str(profile),
        init_expert_location="trivial",
    )
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0),
    )

    with pytest.raises(ValueError, match="no recorded routes"):
        kt_ep_wrapper._build_gpu_expert_masks(server_args)


def test_create_config_uses_server_args_model_config(monkeypatch):
    hf_config = SimpleNamespace(
        num_hidden_layers=6,
        num_hash_layers=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=4,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=hf_config),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: model_config,
        kt_weight_path="/weights",
        enable_eplb=False,
        kt_method="MXFP4",
        kt_mxfp4_backend="amx",
        kt_mxfp4_amx_min_tokens_per_expert=4,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=2,
        kt_expert_placement_strategy="uniform",
        kt_expert_frequency_file=None,
        init_expert_location="trivial",
        kt_threadpool_count=1,
        kt_cpuinfer=8,
        kt_numa_nodes=[0],
        chunked_prefill_size=4096,
        kt_max_deferred_experts_per_token=None,
    )
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0),
    )
    monkeypatch.setattr(kt_ep_wrapper.dist, "is_initialized", lambda: False)

    config = create_kt_config_from_server_args(server_args, layer_idx=2)

    assert config is not None
    assert config.num_layers == 6
    assert config.method == "MXFP4"
    assert config.gpu_prefill_token_threshold == 0
    assert config.mxfp4_prefill_slots == "auto"
    assert config.gpu_experts_mask.tolist() == [True, True, False, False]
