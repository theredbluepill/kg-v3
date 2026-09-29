# AOT ID: ['0_backward']
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/24/c244twrljxxmnikh33ptf5hofp5wqrohbic6givcej4ga6ajnile.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %sum_1 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {dtype: torch.float32})
#   return %buf2
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 256
    x0 = (xindex % 256)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x1*((31 + ks0) // 32)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 256*x1*((31 + ks0) // 32)), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/yg/cygzuessbfmj7cbisc65lsdtuifhydudqctwn5fgdziuc77thfah.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf2 : Tensor "f32[1, 256, 32][8192, 1, 256]cuda:0" = PlaceHolder[target=buf2]
#   %sum_1 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_1
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 34816, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/n3/cn3azjgeqw3agmkhtzzvs4wa77c6rf2qa5voamrnglpjcbsnyu5e.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf1 : Tensor "bf16[256, 512][512, 1]cuda:0" = PlaceHolder[target=buf1]
#   %convert_element_type_18 : Tensor "f32[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
#   return %convert_element_type_18
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/c2/cc2f7sv6d6qimyb4umdxi3ydn4pwu6i5lsiqmmb27h7kpi374q3t.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.mm, aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_6, convert_element_type_6, erf, mul_5
# Graph fragment:
#   %tangents_1 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_2 : Tensor "bf16[256, 512][512, 1]cuda:0" = PlaceHolder[target=permute_2]
#   %mm : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=mm]
#   %addmm : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=addmm]
#   %mm : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%tangents_1, %permute_2), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm, torch.float32), kwargs = {})
#   %mul_5 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_6 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_12 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.5), kwargs = {})
#   %mul_13 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %convert_element_type_6), kwargs = {})
#   %mul_14 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, -0.5), kwargs = {})
#   %exp : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_14,), kwargs = {})
#   %mul_15 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_16 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %mul_15), kwargs = {})
#   %add_14 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_16), kwargs = {})
#   %mul_17 : Tensor "f32[s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %add_14), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[s77, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   return %mm,%convert_element_type_22
triton_tem_fused_gelu_gelu_backward_mm_3 = async_compile.triton('triton_tem_fused_gelu_gelu_backward_mm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_gelu_gelu_backward_mm_3', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused_gelu_gelu_backward_mm_3(arg_A, arg_B, in_ptr2, out_ptr1, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
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
    stride_bk = 512
    stride_bn = 1

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
        b = tl.load(B + (xindex))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp1 = tl.load(in_ptr2 + (tl.broadcast_to(xindex, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = acc.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp2
    tmp11 = -0.5
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.exp(tmp12)
    tmp14 = 0.3989422804014327
    tmp15 = tmp13 * tmp14
    tmp16 = tmp2 * tmp15
    tmp17 = tmp9 + tmp16
    tmp18 = tmp0 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp19, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/ad/cad7io23lphlfp7zl4absqxlelb4nmmyhoqbwupy5b7ji4shom6u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_22 : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_22]
#   %sum_2 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_22, [0], True), kwargs = {dtype: torch.float32})
#   return %buf8
triton_red_fused_sum_4 = async_compile.triton('triton_red_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_sum_4(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 512
    x0 = (xindex % 512)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x1*((31 + ks0) // 32)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 512*x1*((31 + ks0) // 32)), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/b7/cb7zziv5gqw7zvmby7bpxltqknj4dbua2tzlpcyc5ewsy5t2orjv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_22 : Tensor "bf16[s77, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_22]
#   %permute_6 : Tensor "bf16[512, 256][256, 1]cuda:0" = PlaceHolder[target=permute_6]
#   %mm_2 : Tensor "bf16[s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm_2 : Tensor "bf16[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_22, %permute_6), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   return %mm_2,%convert_element_type_28
triton_tem_fused__to_copy_mm_5 = async_compile.triton('triton_tem_fused__to_copy_mm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr1': '*fp32', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_mm_5', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_mm_5(arg_A, arg_B, out_ptr1, ks0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
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
    stride_bk = 256
    stride_bn = 1

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
        b = tl.load(B + (xindex))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = acc.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp0, mask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/mlpbwd_256_512_256/bq/cbqzdt53vej57goepfwm2f644fic2exkau74auzisjaxbfoboeao.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf8 : Tensor "f32[1, 512, 32][16384, 1, 512]cuda:0" = PlaceHolder[target=buf8]
#   %sum_2 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_22, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_2
triton_per_fused_sum_6 = async_compile.triton('triton_per_fused_sum_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 69632, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
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
        primals_3, convert_element_type_2, addmm, convert_element_type_7, permute_2, permute_6, tangents_1 = args
        args.clear()
        s77 = primals_3
        assert_size_stride(convert_element_type_2, (s77, 256), (256, 1))
        assert_size_stride(addmm, (s77, 512), (512, 1))
        assert_size_stride(convert_element_type_7, (s77, 512), (512, 1))
        assert_size_stride(permute_2, (256, 512), (512, 1))
        assert_size_stride(permute_6, (512, 256), (256, 1))
        assert_size_stride(tangents_1, (s77, 256), (256, 1))
        _xnumel = 512*s77
        _xnumel = 256*s77
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((1, 256, 32), (8192, 1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            triton_red_fused_sum_0_r0_numel = (31 + s77) // 32
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_0.run(tangents_1, buf2, s77, 8192, triton_red_fused_sum_0_r0_numel, stream=stream0)
            buf3 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_sum_1.run(buf2, buf3, 256, 32, stream=stream0)
            del buf2
            buf1 = empty_strided_cuda((256, 512), (512, 1), torch.bfloat16)
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.mm(reinterpret_tensor(tangents_1, (256, s77), (1, 256), 0), convert_element_type_7, out=buf1)
            del convert_element_type_7
            buf4 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1, buf4, 131072, stream=stream0)
            del buf1
            buf5 = empty_strided_cuda((s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.mm, aten.gelu_backward, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_tem_fused_gelu_gelu_backward_mm_3.run(tangents_1, permute_2, addmm, buf5, s77, 8*((63 + s77) // 64), 1, 1, stream=stream0)
            del addmm
            del permute_2
            del tangents_1
            buf7 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.mm(reinterpret_tensor(buf5, (512, s77), (1, 512), 0), convert_element_type_2, out=buf7)
            del convert_element_type_2
            buf8 = empty_strided_cuda((1, 512, 32), (16384, 1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            triton_red_fused_sum_4_r0_numel = (31 + s77) // 32
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_4.run(buf5, buf8, s77, 16384, triton_red_fused_sum_4_r0_numel, stream=stream0)
            buf10 = empty_strided_cuda((s77, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_mm_5.run(buf5, permute_6, buf10, s77, 2*((63 + s77) // 64), 1, 1, stream=stream0)
            del buf5
            del permute_6
            buf9 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_sum_6.run(buf8, buf9, 512, 32, stream=stream0)
            del buf8
            buf11 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf7, buf11, 131072, stream=stream0)
            del buf7
        return (buf11, reinterpret_tensor(buf9, (512, ), (1, ), 0), None, buf10, buf4, reinterpret_tensor(buf3, (256, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = 4096
    convert_element_type_2 = rand_strided((4096, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_7 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_6 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((4096, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_3, convert_element_type_2, addmm, convert_element_type_7, permute_2, permute_6, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
