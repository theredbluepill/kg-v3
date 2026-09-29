# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/py/cpyvd4jlpuz4gswbqgwbphkjfbonq2egicetp5ecweygnpq7e2ol.py
# Topologically Sorted Source Nodes: [layer_norm, linear, linear_1, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add_3, add_4, convert_element_type, mul_4, mul_5, rsqrt, sub_1, var_mean
#   linear => convert_element_type_3
#   linear_1 => convert_element_type_9
#   linear_2 => convert_element_type_15
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %getitem_1 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_4 : Tensor "f32[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_4]
#   %convert_element_type : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add_3 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_4 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_5 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg0_1), kwargs = {})
#   %add_4 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg1_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf1,%add_4,%convert_element_type_3,%convert_element_type_9,%convert_element_type_15
triton_per_fused__to_copy_native_layer_norm_0 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tmp1 - tmp11
    tmp19 = 256.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp29, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp29, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/d4/cd4u77doxblprlb2idkst6ejll3qi3ai2mxxtoa5f7ueehsfkyd4.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %arg4_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type_2 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/la/cla3aqwlqv47raqylfdjrcythgxczshgtvo5ntapixtfuulyc2f5.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %arg5_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_1 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/th/cthqppfknorkffps5pskss575yr55nohezx653pqjdierb6x6e65.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear => addmm, convert_element_type_1, convert_element_type_2, convert_element_type_3, permute
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type_3 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_2 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %convert_element_type_1 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_2, [1, 0]), kwargs = {})
#   %addmm : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_1, %convert_element_type_3, %permute), kwargs = {})
#   return %addmm
triton_tem_fused__to_copy_addmm_t_3 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_3', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_3(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/fy/cfyqutaio65djzkqdzpu43pw7lpjm7tdab2foalh3v2udaex7gmu.py
# Topologically Sorted Source Nodes: [reshape, linear_3], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_3 => convert_element_type_20, mm_default_23, permute_3
#   reshape => view_3
# Graph fragment:
#   %getitem_2 : Tensor "bf16[s77, 8, 32][256, 32, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %convert_element_type_20 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_20]
#   %view_3 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [%arg2_1, -1]), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %permute_3 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_20, [1, 0]), kwargs = {})
#   %mm_default_23 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute_3), kwargs = {})
#   return %mm_default_23
triton_tem_fused__to_copy_addmm_t_view_4 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_view_4', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_view_4(arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/cf/ccfdacgsbmrl2nez36lrimsqhu2g6kr2unepf7754imstnntvsdu.py
# Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_64, add_65, convert_element_type_24, mul_46, mul_47, rsqrt_1, sub_20, var_mean_1
#   linear_3 => add_tensor_23, convert_element_type_19
#   linear_4 => convert_element_type_27
#   x => add_57
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_23 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg14_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %getitem_8 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=getitem_8]
#   %buf25 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf25]
#   %arg15_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %add_57 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %add_tensor_23), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_24, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_20 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_24, %getitem_8), kwargs = {})
#   %add_64 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_7, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_64,), kwargs = {})
#   %mul_46 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_1), kwargs = {})
#   %mul_47 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %arg15_1), kwargs = {})
#   %add_65 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %arg16_1), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_65, torch.bfloat16), kwargs = {})
#   return %getitem_8,%buf25,%convert_element_type_27
triton_per_fused__to_copy_add_addmm_native_layer_norm_5 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp6 - tmp16
    tmp24 = 256.0
    tmp25 = (tmp22 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp34, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/zv/czvqmm2tozis4izbcgxt5wkn55h65oojqbakbzcnrrpjq3ipqq46.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_4 => convert_element_type_26
# Graph fragment:
#   %arg17_1 : Tensor "f32[512, 256][256, 1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_26
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/qh/cqhmgs3zg7lulmwnpgubvry54hv4pdlz4hgfoiflj7j346j3zry2.py
# Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
# Source node to ATen node mapping:
#   layer_norm_1 => add_64, add_65, convert_element_type_24, mul_46, mul_47, rsqrt_1, sub_20, var_mean_1
#   linear_3 => add_tensor_23, convert_element_type_19
#   linear_4 => convert_element_type_26, convert_element_type_27, mm_default_22, permute_4
#   x => add_57
# Graph fragment:
#   %convert_element_type_27 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_27]
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_26]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %add_57 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %add_tensor_23), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_24, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_20 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_24, %getitem_8), kwargs = {})
#   %add_64 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_7, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_64,), kwargs = {})
#   %mul_46 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_1), kwargs = {})
#   %mul_47 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %arg15_1), kwargs = {})
#   %add_65 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %arg16_1), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_65, torch.bfloat16), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %permute_4 : Tensor "bf16[256, 512][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_26, [1, 0]), kwargs = {})
#   %mm_default_22 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_27, %permute_4), kwargs = {})
#   return %mm_default_22
triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7(arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B

    M = ks0
    N = 512
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/ik/cik3qdzrpc74ha45kdngtvgque27hagywfuvbbo4x7s7uku6r6ts.py
# Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_81, convert_element_type_31, convert_element_type_32, erf, mul_56, mul_57, mul_58
#   linear_4 => add_tensor_22, convert_element_type_25
# Graph fragment:
#   %mm_default_22 : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_22]
#   %arg18_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %convert_element_type_25 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %add_tensor_22 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %convert_element_type_25), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_22, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.5), kwargs = {})
#   %mul_57 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_57,), kwargs = {})
#   %add_81 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_58 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %add_81), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_58, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32
triton_poi_fused__to_copy_addmm_gelu_8 = async_compile.triton('triton_poi_fused__to_copy_addmm_gelu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4294967296}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i64', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_gelu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_gelu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/pp/cpp276f4qwdpcvl6p5b2x4zquaquqckf35vrb2ucrdww5knh5mrn.py
# Topologically Sorted Source Nodes: [linear_4, gelu, linear_5], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   gelu => add_81, convert_element_type_31, convert_element_type_32, erf, mul_56, mul_57, mul_58
#   linear_4 => add_tensor_22, convert_element_type_25
#   linear_5 => convert_element_type_34, mm_default_21, permute_5
# Graph fragment:
#   %convert_element_type_32 : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_32]
#   %convert_element_type_34 : Tensor "bf16[256, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_34]
#   %convert_element_type_25 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %add_tensor_22 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %convert_element_type_25), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_22, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.5), kwargs = {})
#   %mul_57 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_57,), kwargs = {})
#   %add_81 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_58 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %add_81), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_58, torch.bfloat16), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %permute_5 : Tensor "bf16[512, 256][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_34, [1, 0]), kwargs = {})
#   %mm_default_21 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_32, %permute_5), kwargs = {})
#   return %mm_default_21
triton_tem_fused__to_copy_addmm_gelu_t_9 = async_compile.triton('triton_tem_fused__to_copy_addmm_gelu_t_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_gelu_t_9', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_gelu_t_9(arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/yy/cyyyvhccigbdkmbicvdwrpehrywqttouqr2fsgtpnxcs66z6mrqc.py
# Topologically Sorted Source Nodes: [linear_3, x, linear_5, x_1, layer_norm_2, linear_6, linear_7, linear_8], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_2 => add_95, add_96, convert_element_type_38, mul_69, mul_70, rsqrt_2, sub_30, var_mean_2
#   linear_3 => add_tensor_23, convert_element_type_19
#   linear_5 => add_tensor_21, convert_element_type_33
#   linear_6 => convert_element_type_41
#   linear_7 => convert_element_type_47
#   linear_8 => convert_element_type_53
#   x => add_57
#   x_1 => add_88
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_23 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg14_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %mm_default_21 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_21]
#   %arg20_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %add_88 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_88]
#   %getitem_10 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf35 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf35]
#   %arg21_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %arg22_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %add_96 : Tensor "f32[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_96]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %add_57 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %add_tensor_23), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   %add_tensor_21 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_21, %convert_element_type_33), kwargs = {})
#   %add_88 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %add_tensor_21), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_88, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_30 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_38, %getitem_10), kwargs = {})
#   %add_95 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_95,), kwargs = {})
#   %mul_69 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt_2), kwargs = {})
#   %mul_70 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg21_1), kwargs = {})
#   %add_96 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg22_1), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_96, torch.bfloat16), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_96, torch.bfloat16), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_96, torch.bfloat16), kwargs = {})
#   return %add_88,%getitem_10,%buf35,%add_96,%convert_element_type_41,%convert_element_type_47,%convert_element_type_53
triton_per_fused__to_copy_add_addmm_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tmp11 - tmp21
    tmp29 = 256.0
    tmp30 = (tmp27 / tmp29)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp10, xmask)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp39, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp39, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/jc/cjcq5hu6zcilm473hwb5gbgfjyqwwy5umcis2kzeo65jeyqtrw7a.py
# Topologically Sorted Source Nodes: [linear_9, x_2, linear_11, x_3, layer_norm_4, linear_12, linear_13, linear_14], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_4 => add_187, add_188, convert_element_type_76, mul_134, mul_135, rsqrt_4, sub_59, var_mean_4
#   linear_11 => add_tensor_18, convert_element_type_71
#   linear_12 => convert_element_type_79
#   linear_13 => convert_element_type_85
#   linear_14 => convert_element_type_91
#   linear_9 => add_tensor_20, convert_element_type_57
#   x_2 => add_149
#   x_3 => add_180
# Graph fragment:
#   %add_88 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_88]
#   %mm_default_20 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_20]
#   %arg30_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %mm_default_18 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_18]
#   %arg36_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %add_180 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_180]
#   %getitem_19 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=getitem_19]
#   %buf69 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf69]
#   %arg37_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %add_188 : Tensor "f32[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_188]
#   %convert_element_type_57 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.bfloat16), kwargs = {})
#   %add_tensor_20 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %convert_element_type_57), kwargs = {})
#   %add_149 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %add_tensor_20), kwargs = {})
#   %convert_element_type_71 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   %add_tensor_18 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %convert_element_type_71), kwargs = {})
#   %add_180 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %add_tensor_18), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_180, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_76, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_59 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_76, %getitem_19), kwargs = {})
#   %add_187 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_187,), kwargs = {})
#   %mul_134 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %rsqrt_4), kwargs = {})
#   %mul_135 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %arg37_1), kwargs = {})
#   %add_188 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %arg38_1), kwargs = {})
#   %convert_element_type_79 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_188, torch.bfloat16), kwargs = {})
#   %convert_element_type_85 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_188, torch.bfloat16), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_188, torch.bfloat16), kwargs = {})
#   return %add_180,%getitem_19,%buf69,%add_188,%convert_element_type_79,%convert_element_type_85,%convert_element_type_91
triton_per_fused__to_copy_add_addmm_native_layer_norm_11 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tmp11 - tmp21
    tmp29 = 256.0
    tmp30 = (tmp27 / tmp29)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp10, xmask)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp39, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp39, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_packed_bypass/fr/cfr4ilislkab5nyqaiyhend7s2r5t5oss7tgk7b4bdxoyo65p5yg.py
# Topologically Sorted Source Nodes: [linear_45, x_14, linear_47, x_15, layer_norm_16], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_16 => add_739, add_740, convert_element_type_304, mul_524, mul_525, rsqrt_16, sub_233, var_mean_16
#   linear_45 => add_tensor_2, convert_element_type_285
#   linear_47 => add_tensor, convert_element_type_299
#   x_14 => add_701
#   x_15 => add_732
# Graph fragment:
#   %add_640 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=add_640]
#   %mm_default_2 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %arg126_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg126_1]
#   %mm_default : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg132_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %convert_element_type_304 : Tensor "f32[s77, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_304]
#   %getitem_73 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=getitem_73]
#   %buf273 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf273]
#   %arg133_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg134_1]
#   %convert_element_type_285 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg126_1, torch.bfloat16), kwargs = {})
#   %add_tensor_2 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %convert_element_type_285), kwargs = {})
#   %add_701 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_640, %add_tensor_2), kwargs = {})
#   %convert_element_type_299 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg132_1, torch.bfloat16), kwargs = {})
#   %add_tensor : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %convert_element_type_299), kwargs = {})
#   %add_732 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_701, %add_tensor), kwargs = {})
#   %convert_element_type_304 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_732, torch.float32), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_304, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_233 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_304, %getitem_73), kwargs = {})
#   %add_739 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-05), kwargs = {})
#   %rsqrt_16 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_739,), kwargs = {})
#   %mul_524 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %rsqrt_16), kwargs = {})
#   %mul_525 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_524, %arg133_1), kwargs = {})
#   %add_740 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_525, %arg134_1), kwargs = {})
#   return %convert_element_type_304,%getitem_73,%buf273,%add_740
triton_per_fused__to_copy_add_addmm_native_layer_norm_12 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tmp11 - tmp21
    tmp29 = 256.0
    tmp30 = (tmp27 / tmp29)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp38, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1 = args
        args.clear()
        s77 = arg2_1
        s2 = arg10_1
        s98 = arg12_1
        assert_size_stride(arg0_1, (256, ), (1, ))
        assert_size_stride(arg1_1, (256, ), (1, ))
        assert_size_stride(arg3_1, (s77, 256), (256, 1))
        assert_size_stride(arg4_1, (256, 256), (256, 1))
        assert_size_stride(arg5_1, (256, ), (1, ))
        assert_size_stride(arg6_1, (256, 256), (256, 1))
        assert_size_stride(arg7_1, (256, ), (1, ))
        assert_size_stride(arg8_1, (256, 256), (256, 1))
        assert_size_stride(arg9_1, (256, ), (1, ))
        assert_size_stride(arg11_1, (s2, ), (1, ))
        assert_size_stride(arg13_1, (256, 256), (256, 1))
        assert_size_stride(arg14_1, (256, ), (1, ))
        assert_size_stride(arg15_1, (256, ), (1, ))
        assert_size_stride(arg16_1, (256, ), (1, ))
        assert_size_stride(arg17_1, (512, 256), (256, 1))
        assert_size_stride(arg18_1, (512, ), (1, ))
        assert_size_stride(arg19_1, (256, 512), (512, 1))
        assert_size_stride(arg20_1, (256, ), (1, ))
        assert_size_stride(arg21_1, (256, ), (1, ))
        assert_size_stride(arg22_1, (256, ), (1, ))
        assert_size_stride(arg23_1, (256, 256), (256, 1))
        assert_size_stride(arg24_1, (256, ), (1, ))
        assert_size_stride(arg25_1, (256, 256), (256, 1))
        assert_size_stride(arg26_1, (256, ), (1, ))
        assert_size_stride(arg27_1, (256, 256), (256, 1))
        assert_size_stride(arg28_1, (256, ), (1, ))
        assert_size_stride(arg29_1, (256, 256), (256, 1))
        assert_size_stride(arg30_1, (256, ), (1, ))
        assert_size_stride(arg31_1, (256, ), (1, ))
        assert_size_stride(arg32_1, (256, ), (1, ))
        assert_size_stride(arg33_1, (512, 256), (256, 1))
        assert_size_stride(arg34_1, (512, ), (1, ))
        assert_size_stride(arg35_1, (256, 512), (512, 1))
        assert_size_stride(arg36_1, (256, ), (1, ))
        assert_size_stride(arg37_1, (256, ), (1, ))
        assert_size_stride(arg38_1, (256, ), (1, ))
        assert_size_stride(arg39_1, (256, 256), (256, 1))
        assert_size_stride(arg40_1, (256, ), (1, ))
        assert_size_stride(arg41_1, (256, 256), (256, 1))
        assert_size_stride(arg42_1, (256, ), (1, ))
        assert_size_stride(arg43_1, (256, 256), (256, 1))
        assert_size_stride(arg44_1, (256, ), (1, ))
        assert_size_stride(arg45_1, (256, 256), (256, 1))
        assert_size_stride(arg46_1, (256, ), (1, ))
        assert_size_stride(arg47_1, (256, ), (1, ))
        assert_size_stride(arg48_1, (256, ), (1, ))
        assert_size_stride(arg49_1, (512, 256), (256, 1))
        assert_size_stride(arg50_1, (512, ), (1, ))
        assert_size_stride(arg51_1, (256, 512), (512, 1))
        assert_size_stride(arg52_1, (256, ), (1, ))
        assert_size_stride(arg53_1, (256, ), (1, ))
        assert_size_stride(arg54_1, (256, ), (1, ))
        assert_size_stride(arg55_1, (256, 256), (256, 1))
        assert_size_stride(arg56_1, (256, ), (1, ))
        assert_size_stride(arg57_1, (256, 256), (256, 1))
        assert_size_stride(arg58_1, (256, ), (1, ))
        assert_size_stride(arg59_1, (256, 256), (256, 1))
        assert_size_stride(arg60_1, (256, ), (1, ))
        assert_size_stride(arg61_1, (256, 256), (256, 1))
        assert_size_stride(arg62_1, (256, ), (1, ))
        assert_size_stride(arg63_1, (256, ), (1, ))
        assert_size_stride(arg64_1, (256, ), (1, ))
        assert_size_stride(arg65_1, (512, 256), (256, 1))
        assert_size_stride(arg66_1, (512, ), (1, ))
        assert_size_stride(arg67_1, (256, 512), (512, 1))
        assert_size_stride(arg68_1, (256, ), (1, ))
        assert_size_stride(arg69_1, (256, ), (1, ))
        assert_size_stride(arg70_1, (256, ), (1, ))
        assert_size_stride(arg71_1, (256, 256), (256, 1))
        assert_size_stride(arg72_1, (256, ), (1, ))
        assert_size_stride(arg73_1, (256, 256), (256, 1))
        assert_size_stride(arg74_1, (256, ), (1, ))
        assert_size_stride(arg75_1, (256, 256), (256, 1))
        assert_size_stride(arg76_1, (256, ), (1, ))
        assert_size_stride(arg77_1, (256, 256), (256, 1))
        assert_size_stride(arg78_1, (256, ), (1, ))
        assert_size_stride(arg79_1, (256, ), (1, ))
        assert_size_stride(arg80_1, (256, ), (1, ))
        assert_size_stride(arg81_1, (512, 256), (256, 1))
        assert_size_stride(arg82_1, (512, ), (1, ))
        assert_size_stride(arg83_1, (256, 512), (512, 1))
        assert_size_stride(arg84_1, (256, ), (1, ))
        assert_size_stride(arg85_1, (256, ), (1, ))
        assert_size_stride(arg86_1, (256, ), (1, ))
        assert_size_stride(arg87_1, (256, 256), (256, 1))
        assert_size_stride(arg88_1, (256, ), (1, ))
        assert_size_stride(arg89_1, (256, 256), (256, 1))
        assert_size_stride(arg90_1, (256, ), (1, ))
        assert_size_stride(arg91_1, (256, 256), (256, 1))
        assert_size_stride(arg92_1, (256, ), (1, ))
        assert_size_stride(arg93_1, (256, 256), (256, 1))
        assert_size_stride(arg94_1, (256, ), (1, ))
        assert_size_stride(arg95_1, (256, ), (1, ))
        assert_size_stride(arg96_1, (256, ), (1, ))
        assert_size_stride(arg97_1, (512, 256), (256, 1))
        assert_size_stride(arg98_1, (512, ), (1, ))
        assert_size_stride(arg99_1, (256, 512), (512, 1))
        assert_size_stride(arg100_1, (256, ), (1, ))
        assert_size_stride(arg101_1, (256, ), (1, ))
        assert_size_stride(arg102_1, (256, ), (1, ))
        assert_size_stride(arg103_1, (256, 256), (256, 1))
        assert_size_stride(arg104_1, (256, ), (1, ))
        assert_size_stride(arg105_1, (256, 256), (256, 1))
        assert_size_stride(arg106_1, (256, ), (1, ))
        assert_size_stride(arg107_1, (256, 256), (256, 1))
        assert_size_stride(arg108_1, (256, ), (1, ))
        assert_size_stride(arg109_1, (256, 256), (256, 1))
        assert_size_stride(arg110_1, (256, ), (1, ))
        assert_size_stride(arg111_1, (256, ), (1, ))
        assert_size_stride(arg112_1, (256, ), (1, ))
        assert_size_stride(arg113_1, (512, 256), (256, 1))
        assert_size_stride(arg114_1, (512, ), (1, ))
        assert_size_stride(arg115_1, (256, 512), (512, 1))
        assert_size_stride(arg116_1, (256, ), (1, ))
        assert_size_stride(arg117_1, (256, ), (1, ))
        assert_size_stride(arg118_1, (256, ), (1, ))
        assert_size_stride(arg119_1, (256, 256), (256, 1))
        assert_size_stride(arg120_1, (256, ), (1, ))
        assert_size_stride(arg121_1, (256, 256), (256, 1))
        assert_size_stride(arg122_1, (256, ), (1, ))
        assert_size_stride(arg123_1, (256, 256), (256, 1))
        assert_size_stride(arg124_1, (256, ), (1, ))
        assert_size_stride(arg125_1, (256, 256), (256, 1))
        assert_size_stride(arg126_1, (256, ), (1, ))
        assert_size_stride(arg127_1, (256, ), (1, ))
        assert_size_stride(arg128_1, (256, ), (1, ))
        assert_size_stride(arg129_1, (512, 256), (256, 1))
        assert_size_stride(arg130_1, (512, ), (1, ))
        assert_size_stride(arg131_1, (256, 512), (512, 1))
        assert_size_stride(arg132_1, (256, ), (1, ))
        assert_size_stride(arg133_1, (256, ), (1, ))
        assert_size_stride(arg134_1, (256, ), (1, ))
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        _xnumel = 512*s77
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf4 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [layer_norm, linear, linear_1, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_0.run(arg3_1, arg0_1, arg1_1, buf4, buf8, buf12, s77, 256, stream=stream0)
            del arg0_1
            del arg1_1
            buf5 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg4_1, buf5, 65536, stream=stream0)
            del arg4_1
            buf6 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg5_1, buf6, 256, stream=stream0)
            del arg5_1
            buf7 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf6, buf4, buf5, buf7, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf9 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg6_1, buf9, 65536, stream=stream0)
            del arg6_1
            buf10 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg7_1, buf10, 256, stream=stream0)
            del arg7_1
            buf11 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf10, buf8, buf9, buf11, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf13 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg8_1, buf13, 65536, stream=stream0)
            del arg8_1
            buf14 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg9_1, buf14, 256, stream=stream0)
            del arg9_1
            buf15 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf14, buf12, buf13, buf15, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            # Topologically Sorted Source Nodes: [q, k, v, _flash_attention_forward], Original ATen: [aten.view, aten._flash_attention_forward]
            buf16 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf7, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf11, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf15, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf17 = buf16[0]
            assert_size_stride(buf17, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf17, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf16
            buf22 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg13_1, buf22, 65536, stream=stream0)
            del arg13_1
            buf23 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [reshape, linear_3], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf17, buf22, buf23, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf27 = reinterpret_tensor(buf17, (s77, 256), (256, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(arg3_1, buf23, arg14_1, arg15_1, arg16_1, buf27, s77, 256, stream=stream0)
            del arg15_1
            del arg16_1
            buf28 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg17_1, buf28, 131072, stream=stream0)
            del arg17_1
            buf29 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf27, buf28, buf29, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf30 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf30, arg18_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg18_1
            buf31 = reinterpret_tensor(buf28, (256, 512), (512, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg19_1, buf31, 131072, stream=stream0)
            del arg19_1
            buf32 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu, linear_5], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf30, buf31, buf32, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf30
            buf33 = buf23; del buf23  # reuse
            buf38 = buf15; del buf15  # reuse
            buf42 = buf11; del buf11  # reuse
            buf46 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_3, x, linear_5, x_1, layer_norm_2, linear_6, linear_7, linear_8], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_10.run(buf33, arg3_1, arg14_1, buf32, arg20_1, arg21_1, arg22_1, buf38, buf42, buf46, s77, 256, stream=stream0)
            del arg14_1
            del arg20_1
            del arg21_1
            del arg22_1
            del arg3_1
            buf39 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg23_1, buf39, 65536, stream=stream0)
            del arg23_1
            buf40 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg24_1, buf40, 256, stream=stream0)
            del arg24_1
            buf41 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf40, buf38, buf39, buf41, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf43 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg25_1, buf43, 65536, stream=stream0)
            del arg25_1
            buf44 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg26_1, buf44, 256, stream=stream0)
            del arg26_1
            buf45 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf44, buf42, buf43, buf45, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg27_1, buf47, 65536, stream=stream0)
            del arg27_1
            buf48 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg28_1, buf48, 256, stream=stream0)
            del arg28_1
            buf49 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf48, buf46, buf47, buf49, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf46
            del buf48
            # Topologically Sorted Source Nodes: [q_1, k_1, v_1, _flash_attention_forward_1], Original ATen: [aten.view, aten._flash_attention_forward]
            buf50 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf41, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf45, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf49, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf51 = buf50[0]
            assert_size_stride(buf51, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf51, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf50
            buf56 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg29_1, buf56, 65536, stream=stream0)
            del arg29_1
            buf57 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [reshape_1, linear_9], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf51, buf56, buf57, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf56
            buf61 = reinterpret_tensor(buf51, (s77, 256), (256, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [linear_9, x_2, layer_norm_3, linear_10], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf33, buf57, arg30_1, arg31_1, arg32_1, buf61, s77, 256, stream=stream0)
            del arg31_1
            del arg32_1
            buf62 = reinterpret_tensor(buf31, (512, 256), (256, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg33_1, buf62, 131072, stream=stream0)
            del arg33_1
            buf63 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, x_2, layer_norm_3, linear_10], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf61, buf62, buf63, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf64 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_10, gelu_1], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf64, arg34_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg34_1
            buf65 = reinterpret_tensor(buf62, (256, 512), (512, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg35_1, buf65, 131072, stream=stream0)
            del arg35_1
            buf66 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [linear_10, gelu_1, linear_11], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf64, buf65, buf66, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf64
            del buf65
            buf67 = buf33; del buf33  # reuse
            buf72 = buf45; del buf45  # reuse
            buf76 = buf41; del buf41  # reuse
            buf80 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, x_2, linear_11, x_3, layer_norm_4, linear_12, linear_13, linear_14], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf67, buf57, arg30_1, buf66, arg36_1, arg37_1, arg38_1, buf72, buf76, buf80, s77, 256, stream=stream0)
            del arg30_1
            del arg36_1
            del arg37_1
            del arg38_1
            del buf57
            buf73 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg39_1, buf73, 65536, stream=stream0)
            del arg39_1
            buf74 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg40_1, buf74, 256, stream=stream0)
            del arg40_1
            buf75 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf74, buf72, buf73, buf75, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf77 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg41_1, buf77, 65536, stream=stream0)
            del arg41_1
            buf78 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg42_1, buf78, 256, stream=stream0)
            del arg42_1
            buf79 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf78, buf76, buf77, buf79, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf81 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg43_1, buf81, 65536, stream=stream0)
            del arg43_1
            buf82 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg44_1, buf82, 256, stream=stream0)
            del arg44_1
            buf83 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf82, buf80, buf81, buf83, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf80
            del buf82
            # Topologically Sorted Source Nodes: [q_2, k_2, v_2, _flash_attention_forward_2], Original ATen: [aten.view, aten._flash_attention_forward]
            buf84 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf75, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf79, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf83, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf85 = buf84[0]
            assert_size_stride(buf85, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf85, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf84
            buf90 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg45_1, buf90, 65536, stream=stream0)
            del arg45_1
            buf91 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [reshape_2, linear_15], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf85, buf90, buf91, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf90
            buf95 = reinterpret_tensor(buf85, (s77, 256), (256, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [linear_15, x_4, layer_norm_5, linear_16], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf67, buf91, arg46_1, arg47_1, arg48_1, buf95, s77, 256, stream=stream0)
            del arg47_1
            del arg48_1
            buf96 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg49_1, buf96, 131072, stream=stream0)
            del arg49_1
            buf97 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_15, x_4, layer_norm_5, linear_16], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf95, buf96, buf97, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf98 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [linear_16, gelu_2], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf98, arg50_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg50_1
            buf99 = reinterpret_tensor(buf96, (256, 512), (512, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg51_1, buf99, 131072, stream=stream0)
            del arg51_1
            buf100 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [linear_16, gelu_2, linear_17], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf98, buf99, buf100, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf98
            del buf99
            buf101 = buf67; del buf67  # reuse
            buf106 = buf79; del buf79  # reuse
            buf110 = buf75; del buf75  # reuse
            buf114 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_15, x_4, linear_17, x_5, layer_norm_6, linear_18, linear_19, linear_20], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf101, buf91, arg46_1, buf100, arg52_1, arg53_1, arg54_1, buf106, buf110, buf114, s77, 256, stream=stream0)
            del arg46_1
            del arg52_1
            del arg53_1
            del arg54_1
            del buf100
            buf107 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg55_1, buf107, 65536, stream=stream0)
            del arg55_1
            buf108 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg56_1, buf108, 256, stream=stream0)
            del arg56_1
            buf109 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf108, buf106, buf107, buf109, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf111 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg57_1, buf111, 65536, stream=stream0)
            del arg57_1
            buf112 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg58_1, buf112, 256, stream=stream0)
            del arg58_1
            buf113 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf112, buf110, buf111, buf113, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf115 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg59_1, buf115, 65536, stream=stream0)
            del arg59_1
            buf116 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg60_1, buf116, 256, stream=stream0)
            del arg60_1
            buf117 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf116, buf114, buf115, buf117, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf114
            del buf116
            # Topologically Sorted Source Nodes: [q_3, k_3, v_3, _flash_attention_forward_3], Original ATen: [aten.view, aten._flash_attention_forward]
            buf118 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf109, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf113, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf117, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf119 = buf118[0]
            assert_size_stride(buf119, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf119, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf118
            buf124 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg61_1, buf124, 65536, stream=stream0)
            del arg61_1
            buf125 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [reshape_3, linear_21], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf119, buf124, buf125, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf124
            buf129 = reinterpret_tensor(buf119, (s77, 256), (256, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [linear_21, x_6, layer_norm_7, linear_22], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf101, buf125, arg62_1, arg63_1, arg64_1, buf129, s77, 256, stream=stream0)
            del arg63_1
            del arg64_1
            buf130 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg65_1, buf130, 131072, stream=stream0)
            del arg65_1
            buf131 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_21, x_6, layer_norm_7, linear_22], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf129, buf130, buf131, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf132 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_22, gelu_3], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf132, arg66_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg66_1
            buf133 = reinterpret_tensor(buf130, (256, 512), (512, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg67_1, buf133, 131072, stream=stream0)
            del arg67_1
            buf134 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_22, gelu_3, linear_23], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf132, buf133, buf134, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf132
            del buf133
            buf135 = buf101; del buf101  # reuse
            buf140 = buf113; del buf113  # reuse
            buf144 = buf109; del buf109  # reuse
            buf148 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_21, x_6, linear_23, x_7, layer_norm_8, linear_24, linear_25, linear_26], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf135, buf125, arg62_1, buf134, arg68_1, arg69_1, arg70_1, buf140, buf144, buf148, s77, 256, stream=stream0)
            del arg62_1
            del arg68_1
            del arg69_1
            del arg70_1
            del buf125
            buf141 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg71_1, buf141, 65536, stream=stream0)
            del arg71_1
            buf142 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg72_1, buf142, 256, stream=stream0)
            del arg72_1
            buf143 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf142, buf140, buf141, buf143, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf145 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg73_1, buf145, 65536, stream=stream0)
            del arg73_1
            buf146 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg74_1, buf146, 256, stream=stream0)
            del arg74_1
            buf147 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf146, buf144, buf145, buf147, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf149 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg75_1, buf149, 65536, stream=stream0)
            del arg75_1
            buf150 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg76_1, buf150, 256, stream=stream0)
            del arg76_1
            buf151 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf150, buf148, buf149, buf151, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf148
            del buf150
            # Topologically Sorted Source Nodes: [q_4, k_4, v_4, _flash_attention_forward_4], Original ATen: [aten.view, aten._flash_attention_forward]
            buf152 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf143, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf147, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf151, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf153 = buf152[0]
            assert_size_stride(buf153, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf153, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf152
            buf158 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg77_1, buf158, 65536, stream=stream0)
            del arg77_1
            buf159 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [reshape_4, linear_27], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf153, buf158, buf159, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf158
            buf163 = reinterpret_tensor(buf153, (s77, 256), (256, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [linear_27, x_8, layer_norm_9, linear_28], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf135, buf159, arg78_1, arg79_1, arg80_1, buf163, s77, 256, stream=stream0)
            del arg79_1
            del arg80_1
            buf164 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg81_1, buf164, 131072, stream=stream0)
            del arg81_1
            buf165 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27, x_8, layer_norm_9, linear_28], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf163, buf164, buf165, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf166 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [linear_28, gelu_4], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf166, arg82_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg82_1
            buf167 = reinterpret_tensor(buf164, (256, 512), (512, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg83_1, buf167, 131072, stream=stream0)
            del arg83_1
            buf168 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [linear_28, gelu_4, linear_29], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf166, buf167, buf168, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf166
            del buf167
            buf169 = buf135; del buf135  # reuse
            buf174 = buf147; del buf147  # reuse
            buf178 = buf143; del buf143  # reuse
            buf182 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27, x_8, linear_29, x_9, layer_norm_10, linear_30, linear_31, linear_32], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf169, buf159, arg78_1, buf168, arg84_1, arg85_1, arg86_1, buf174, buf178, buf182, s77, 256, stream=stream0)
            del arg78_1
            del arg84_1
            del arg85_1
            del arg86_1
            del buf159
            buf175 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg87_1, buf175, 65536, stream=stream0)
            del arg87_1
            buf176 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg88_1, buf176, 256, stream=stream0)
            del arg88_1
            buf177 = buf168; del buf168  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf176, buf174, buf175, buf177, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf179 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg89_1, buf179, 65536, stream=stream0)
            del arg89_1
            buf180 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg90_1, buf180, 256, stream=stream0)
            del arg90_1
            buf181 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf180, buf178, buf179, buf181, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf183 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg91_1, buf183, 65536, stream=stream0)
            del arg91_1
            buf184 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg92_1, buf184, 256, stream=stream0)
            del arg92_1
            buf185 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf184, buf182, buf183, buf185, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf182
            del buf184
            # Topologically Sorted Source Nodes: [q_5, k_5, v_5, _flash_attention_forward_5], Original ATen: [aten.view, aten._flash_attention_forward]
            buf186 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf177, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf181, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf185, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf187 = buf186[0]
            assert_size_stride(buf187, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf187, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf186
            buf192 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg93_1, buf192, 65536, stream=stream0)
            del arg93_1
            buf193 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [reshape_5, linear_33], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf187, buf192, buf193, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf192
            buf197 = reinterpret_tensor(buf187, (s77, 256), (256, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [linear_33, x_10, layer_norm_11, linear_34], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf169, buf193, arg94_1, arg95_1, arg96_1, buf197, s77, 256, stream=stream0)
            del arg95_1
            del arg96_1
            buf198 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg97_1, buf198, 131072, stream=stream0)
            del arg97_1
            buf199 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33, x_10, layer_norm_11, linear_34], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf197, buf198, buf199, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf200 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [linear_34, gelu_5], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf200, arg98_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg98_1
            buf201 = reinterpret_tensor(buf198, (256, 512), (512, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg99_1, buf201, 131072, stream=stream0)
            del arg99_1
            buf202 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [linear_34, gelu_5, linear_35], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf200, buf201, buf202, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf200
            del buf201
            buf203 = buf169; del buf169  # reuse
            buf208 = buf181; del buf181  # reuse
            buf212 = buf177; del buf177  # reuse
            buf216 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33, x_10, linear_35, x_11, layer_norm_12, linear_36, linear_37, linear_38], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf203, buf193, arg94_1, buf202, arg100_1, arg101_1, arg102_1, buf208, buf212, buf216, s77, 256, stream=stream0)
            del arg100_1
            del arg101_1
            del arg102_1
            del arg94_1
            del buf193
            buf209 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg103_1, buf209, 65536, stream=stream0)
            del arg103_1
            buf210 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg104_1, buf210, 256, stream=stream0)
            del arg104_1
            buf211 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf210, buf208, buf209, buf211, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf213 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg105_1, buf213, 65536, stream=stream0)
            del arg105_1
            buf214 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg106_1, buf214, 256, stream=stream0)
            del arg106_1
            buf215 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf214, buf212, buf213, buf215, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf217 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg107_1, buf217, 65536, stream=stream0)
            del arg107_1
            buf218 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg108_1, buf218, 256, stream=stream0)
            del arg108_1
            buf219 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf218, buf216, buf217, buf219, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf216
            del buf218
            # Topologically Sorted Source Nodes: [q_6, k_6, v_6, _flash_attention_forward_6], Original ATen: [aten.view, aten._flash_attention_forward]
            buf220 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf211, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf215, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf219, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            buf221 = buf220[0]
            assert_size_stride(buf221, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf221, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf220
            buf226 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg109_1, buf226, 65536, stream=stream0)
            del arg109_1
            buf227 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [reshape_6, linear_39], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf221, buf226, buf227, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf226
            buf231 = reinterpret_tensor(buf221, (s77, 256), (256, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [linear_39, x_12, layer_norm_13, linear_40], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf203, buf227, arg110_1, arg111_1, arg112_1, buf231, s77, 256, stream=stream0)
            del arg111_1
            del arg112_1
            buf232 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg113_1, buf232, 131072, stream=stream0)
            del arg113_1
            buf233 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39, x_12, layer_norm_13, linear_40], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf231, buf232, buf233, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf234 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [linear_40, gelu_6], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf234, arg114_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg114_1
            buf235 = reinterpret_tensor(buf232, (256, 512), (512, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg115_1, buf235, 131072, stream=stream0)
            del arg115_1
            buf236 = buf231; del buf231  # reuse
            # Topologically Sorted Source Nodes: [linear_40, gelu_6, linear_41], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf234, buf235, buf236, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf234
            del buf235
            buf237 = buf203; del buf203  # reuse
            buf242 = buf215; del buf215  # reuse
            buf246 = buf211; del buf211  # reuse
            buf250 = empty_strided_cuda((s77, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39, x_12, linear_41, x_13, layer_norm_14, linear_42, linear_43, linear_44], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_11.run(buf237, buf227, arg110_1, buf236, arg116_1, arg117_1, arg118_1, buf242, buf246, buf250, s77, 256, stream=stream0)
            del arg110_1
            del arg116_1
            del arg117_1
            del arg118_1
            del buf227
            buf243 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg119_1, buf243, 65536, stream=stream0)
            del arg119_1
            buf244 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg120_1, buf244, 256, stream=stream0)
            del arg120_1
            buf245 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf244, buf242, buf243, buf245, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf247 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg121_1, buf247, 65536, stream=stream0)
            del arg121_1
            buf248 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg122_1, buf248, 256, stream=stream0)
            del arg122_1
            buf249 = buf242; del buf242  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf248, buf246, buf247, buf249, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            buf251 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg123_1, buf251, 65536, stream=stream0)
            del arg123_1
            buf252 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg124_1, buf252, 256, stream=stream0)
            del arg124_1
            buf253 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_3.run(buf252, buf250, buf251, buf253, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf250
            del buf252
            # Topologically Sorted Source Nodes: [q_7, k_7, v_7, _flash_attention_forward_7], Original ATen: [aten.view, aten._flash_attention_forward]
            buf254 = torch.ops.aten._flash_attention_forward.default(reinterpret_tensor(buf245, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf249, (s77, 8, 32), (256, 32, 1), 0), reinterpret_tensor(buf253, (s77, 8, 32), (256, 32, 1), 0), arg11_1, arg11_1, s98, s98, 0.0, False, False)
            del arg11_1
            del buf245
            del buf249
            buf255 = buf254[0]
            assert_size_stride(buf255, (s77, 8, 32), (256, 32, 1), 'torch.ops.aten._flash_attention_forward.default')
            assert_alignment(buf255, 16, 'torch.ops.aten._flash_attention_forward.default')
            del buf254
            buf260 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg125_1, buf260, 65536, stream=stream0)
            del arg125_1
            buf261 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [reshape_7, linear_45], Original ATen: [aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_view_4.run(buf255, buf260, buf261, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf260
            buf265 = reinterpret_tensor(buf255, (s77, 256), (256, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [linear_45, x_14, layer_norm_15, linear_46], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_5.run(buf237, buf261, arg126_1, arg127_1, arg128_1, buf265, s77, 256, stream=stream0)
            del arg127_1
            del arg128_1
            buf266 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg129_1, buf266, 131072, stream=stream0)
            del arg129_1
            buf267 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_45, x_14, layer_norm_15, linear_46], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_7.run(buf265, buf266, buf267, s77, 4*((127 + s77) // 128), 1, 1, stream=stream0)
            buf268 = buf267; del buf267  # reuse
            # Topologically Sorted Source Nodes: [linear_46, gelu_7], Original ATen: [aten._to_copy, aten.addmm, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_8_xnumel = 512*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_8.run(buf268, arg130_1, triton_poi_fused__to_copy_addmm_gelu_8_xnumel, stream=stream0)
            del arg130_1
            buf269 = reinterpret_tensor(buf266, (256, 512), (512, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg131_1, buf269, 131072, stream=stream0)
            del arg131_1
            buf270 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [linear_46, gelu_7, linear_47], Original ATen: [aten._to_copy, aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_9.run(buf268, buf269, buf270, s77, 2*((127 + s77) // 128), 1, 1, stream=stream0)
            del buf268
            del buf269
            buf271 = empty_strided_cuda((s77, 256), (256, 1), torch.float32)
            buf275 = buf271; del buf271  # reuse
            # Topologically Sorted Source Nodes: [linear_45, x_14, linear_47, x_15, layer_norm_16], Original ATen: [aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_12.run(buf275, buf237, buf261, arg126_1, buf270, arg132_1, arg133_1, arg134_1, s77, 256, stream=stream0)
            del arg126_1
            del arg132_1
            del arg133_1
            del arg134_1
            del buf237
            del buf261
            del buf270
        return (buf275, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = 4194305
    arg3_1 = rand_strided((4194305, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = 5917
    arg11_1 = rand_strided((5917, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg12_1 = 709
    arg13_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
