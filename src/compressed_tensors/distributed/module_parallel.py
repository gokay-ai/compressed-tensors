# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.distributed.assign import greedy_bin_packing
from compressed_tensors.distributed.utils import as_broadcastable, set_source_process
from compressed_tensors.offload.utils import as_single_threaded, module_size
from compressed_tensors.utils.module import (
    get_direct_state_dict,
    replace_direct_state_dict,
)
from tqdm import tqdm


__all__ = ["replace_module_parallel"]

T = TypeVar("T", bound=torch.nn.Module)


def _broadcast_direct_state_dict(module: torch.nn.Module, src: int) -> None:
    """Broadcast direct module tensors without pickling tensor storage.

    Object collectives pickle tensors, and FP8 tensor storage fails to round-trip
    through PyTorch's legacy loader. Non-source ranks already have matching meta
    tensors, so broadcast only their source device types before sending tensor data.
    """
    rank = dist.get_rank()
    state_dict = get_direct_state_dict(module)
    if rank == src:
        tensor_metadata = [
            (name, None if tensor is None else tensor.device.type)
            for name, tensor in state_dict.items()
        ]
    else:
        tensor_metadata = None

    tensor_metadata_list = [tensor_metadata]
    dist.broadcast_object_list(tensor_metadata_list, src=src)
    tensor_metadata = tensor_metadata_list[0]

    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    elif "xccl" in backend:
        device = torch.device("xpu", torch.accelerator.current_device_index())
    else:
        device = torch.device("cpu")

    received_state_dict = {}
    for name, tensor_device_type in tensor_metadata:
        tensor = state_dict[name]
        if tensor_device_type is None:
            assert tensor is None
            received_state_dict[name] = None
            continue

        assert tensor is not None
        if tensor_device_type in ("cuda", "xpu"):
            tensor_device = torch.device(
                tensor_device_type, torch.accelerator.current_device_index()
            )
        else:
            tensor_device = torch.device(tensor_device_type)

        if rank == src:
            payload = tensor.detach().to(device).contiguous()
        else:
            payload = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

        if payload.numel() > 0:
            dist.broadcast(as_broadcastable(payload), src=src)

        if rank != src:
            received_tensor = payload.to(tensor_device)
            if tuple(received_tensor.stride()) != tuple(tensor.stride()):
                strided_tensor = torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor_device,
                )
                strided_tensor.copy_(received_tensor)
                received_tensor = strided_tensor
            received_state_dict[name] = received_tensor

    if rank != src:
        replace_direct_state_dict(module, received_state_dict)


def replace_module_parallel(
    modules: list[T],
    apply_fn: Callable[[T], None],
    weight_fn: Callable[[T], int | float] = module_size,
    desc: Optional[str] = None,
):
    """Apply a function to modules in parallel across distributed ranks.

    Distributes modules across ranks using greedy bin packing, then applies
    the function to each module on its assigned rank. Non-processing ranks
    temporarily move their modules to meta device to avoid increasing peak
    memory usage during compression.

    This implements the 4-step algorithm:
    1. Decouple: Move non-processing rank modules to meta device
    2. Compress On Meta: Apply function on meta device (prepare for step 4)
    3. Compress On Device: Processing rank applies function without sync
    4. Recouple: Broadcast offload pointer information across ranks

    :param modules: list of modules to process
    :param apply_fn: function to apply to each module
    :param weight_fn: function that returns the weight/size of a module
        for load balancing across ranks
    :param desc: optional description for the progress bar (shown on each rank)
    """
    from compressed_tensors.offload import OffloadCache, disable_onloading, to_meta

    _, _, assigned_rank = greedy_bin_packing(modules, dist.get_world_size(), weight_fn)

    # Count modules assigned to this rank and create progress bar
    rank = dist.get_rank()
    num_assigned = sum(int(a == rank) for a in assigned_rank.values())
    progress = tqdm(
        total=num_assigned, desc=desc, position=rank, disable=(desc is None)
    )

    # Step 1 & 2: Decouple and compress on meta for non-processing ranks
    with disable_onloading():
        for module in modules:
            if assigned_rank[module] != dist.get_rank():
                to_meta(module)  # 1. remove non-processing rank pointers
                apply_fn(module)  # 2. compress on meta to match state dict for step 4

    # Step 3: Apply on device for processing rank
    with as_single_threaded():
        for module in modules:
            if assigned_rank[module] == dist.get_rank():
                apply_fn(module)  # 3. compress without triggering sync
                progress.update(1)

    # Step 4: Recouple - broadcast source offload across ranks
    for module in modules:
        if isinstance(module._parameters, OffloadCache):
            assert isinstance(module._buffers, OffloadCache)
            with set_source_process(assigned_rank[module]):
                module._parameters.offloaded_values = {
                    name: module._parameters.offload(value)
                    for name, value in module._parameters.offloaded_values.items()
                }
                module._buffers.offloaded_values = {
                    name: module._buffers.offload(value)
                    for name, value in module._buffers.offloaded_values.items()
                }

        else:
            _broadcast_direct_state_dict(module, assigned_rank[module])
