# AOT ID: ['1_inference']
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/fl/cfllbxggdwveroybslcszht3z4g6bdfuhw54ansrr4fwgedvelxw.py
# Topologically Sorted Source Nodes: [layer_norm, linear, linear_1, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add_4, add_5, convert_element_type, mul_7, mul_8, rsqrt, sub_2, var_mean
#   linear => convert_element_type_3
#   linear_1 => convert_element_type_9
#   linear_2 => convert_element_type_15
# Graph fragment:
#   %arg4_1 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %getitem_1 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_5 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_5]
#   %convert_element_type : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add_4 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_7 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_8 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %arg0_1), kwargs = {})
#   %add_5 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %arg1_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf1,%add_5,%convert_element_type_3,%convert_element_type_9,%convert_element_type_15
triton_per_fused__to_copy_native_layer_norm_0 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/d4/cd4u77doxblprlb2idkst6ejll3qi3ai2mxxtoa5f7ueehsfkyd4.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %arg6_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_2 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/la/cla3aqwlqv47raqylfdjrcythgxczshgtvo5ntapixtfuulyc2f5.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %arg7_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_1 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/yi/cyi6se4hbpzz4b3prxysi6s6njf54avto4atap3zlrngp7q7yjyc.py
# Topologically Sorted Source Nodes: [linear, q, transpose, linear_1, k, transpose_1, linear_2, v, transpose_2, getitem_3, attn, linear_6, q_1, transpose_4, linear_7, k_1, transpose_5, linear_8, v_1, transpose_6, getitem_7, attn_2, linear_12, q_2, transpose_8, linear_13, k_2, transpose_9, linear_14, v_2, transpose_10, getitem_11, attn_4], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn => _scaled_dot_product_efficient_attention, constant_pad_nd, expand, full_default, full_default_1, slice_3, where
#   attn_2 => _scaled_dot_product_efficient_attention_1, constant_pad_nd_1, expand_1, full_default_2, full_default_3, slice_6, where_1
#   attn_4 => _scaled_dot_product_efficient_attention_2, constant_pad_nd_2, expand_2, full_default_4, full_default_5, slice_9, where_2
#   getitem_11 => unsqueeze_4, unsqueeze_5
#   getitem_3 => unsqueeze, unsqueeze_1
#   getitem_7 => unsqueeze_2, unsqueeze_3
#   k => view_5
#   k_1 => view_21
#   k_2 => view_37
#   linear => view_1
#   linear_1 => view_4
#   linear_12 => view_33
#   linear_13 => view_36
#   linear_14 => view_39
#   linear_2 => view_7
#   linear_6 => view_17
#   linear_7 => view_20
#   linear_8 => view_23
#   q => view_2
#   q_1 => view_18
#   q_2 => view_34
#   transpose => permute_3
#   transpose_1 => permute_4
#   transpose_10 => permute_25
#   transpose_2 => permute_5
#   transpose_4 => permute_13
#   transpose_5 => permute_14
#   transpose_6 => permute_15
#   transpose_8 => permute_23
#   transpose_9 => permute_24
#   v => view_8
#   v_1 => view_24
#   v_2 => view_40
# Graph fragment:
#   %arg5_1 : Tensor "b8[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %view_1 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_2 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_3 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_5 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_4 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %view_7 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_8 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_5 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze : Tensor "b8[s77, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "b8[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_1, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[s77, 1, 1, s27 - (Mod(s27, 8)) + 8][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_49], 0.0), kwargs = {})
#   %slice_3 : Tensor "bf16[s77, 1, 1, s27][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %arg3_1), kwargs = {})
#   %expand : Tensor "bf16[s77, 8, s27, s27][Max(1, s27 - (Mod(s27, 8)) + 8), 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_3, [%arg2_1, 8, %arg3_1, %arg3_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_3, %permute_4, %permute_5, %expand, False), kwargs = {})
#   %view_17 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_6, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_18 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_17, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_13 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %view_20 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_21 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_14 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %view_23 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_24 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_15 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_24, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_2 : Tensor "b8[s77, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "b8[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 2), kwargs = {})
#   %full_default_3 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "bf16[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_3, %full_default_3, %full_default_2), kwargs = {})
#   %constant_pad_nd_1 : Tensor "bf16[s77, 1, 1, s27 - (Mod(s27, 8)) + 8][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where_1, [0, %sub_49], 0.0), kwargs = {})
#   %slice_6 : Tensor "bf16[s77, 1, 1, s27][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_1, -1, 0, %arg3_1), kwargs = {})
#   %expand_1 : Tensor "bf16[s77, 8, s27, s27][Max(1, s27 - (Mod(s27, 8)) + 8), 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_6, [%arg2_1, 8, %arg3_1, %arg3_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_13, %permute_14, %permute_15, %expand_1, False), kwargs = {})
#   %view_33 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_34 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_33, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_23 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_34, [0, 2, 1, 3]), kwargs = {})
#   %view_36 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_37 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_36, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_24 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1, 3]), kwargs = {})
#   %view_39 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_14, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_40 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_25 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_40, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_4 : Tensor "b8[s77, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %unsqueeze_5 : Tensor "b8[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %full_default_5 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_4 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "bf16[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_5, %full_default_5, %full_default_4), kwargs = {})
#   %constant_pad_nd_2 : Tensor "bf16[s77, 1, 1, s27 - (Mod(s27, 8)) + 8][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where_2, [0, %sub_49], 0.0), kwargs = {})
#   %slice_9 : Tensor "bf16[s77, 1, 1, s27][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_2, -1, 0, %arg3_1), kwargs = {})
#   %expand_2 : Tensor "bf16[s77, 8, s27, s27][Max(1, s27 - (Mod(s27, 8)) + 8), 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_9, [%arg2_1, 8, %arg3_1, %arg3_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_23, %permute_24, %permute_25, %expand_2, False), kwargs = {})
#   return %buf16,%buf50,%buf84
triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x2 = xindex // ks1
    x4 = xindex // ks0
    tmp0 = x0
    tmp1 = ks0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + ks0*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = 0.0
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tl.store(out_ptr0 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
    tl.store(out_ptr1 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
    tl.store(out_ptr2 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/cf/ccfsw7cakj3gvmpsokdxritqd5zh6qtoasajsew5rnwr75q2hf2x.py
# Topologically Sorted Source Nodes: [attn_1, reshape, linear_3], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   attn_1 => permute_6
#   linear_3 => convert_element_type_20, mm_default_23, permute_7, view_10
#   reshape => view_9
# Graph fragment:
#   %getitem_2 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %convert_element_type_20 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_20]
#   %permute_6 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_9 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [%arg2_1, %arg3_1, -1]), kwargs = {})
#   %view_10 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [%mul_25, 256]), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_20, [1, 0]), kwargs = {})
#   %mm_default_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_10, %permute_7), kwargs = {})
#   return %mm_default_23
triton_tem_fused__to_copy_addmm_t_transpose_view_4 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_transpose_view_4', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_transpose_view_4(arg_A, arg_B, out_ptr0, ks0, ks1):
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

    M = ks0*ks1
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/65/c65rzv7b3znom7q6e5nieq5povzsim3vd6pv5ecwibfdgcr4mvid.py
# Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_162, add_163, convert_element_type_24, mul_150, mul_151, rsqrt_1, sub_72, var_mean_1
#   linear_3 => add_tensor_23, convert_element_type_19, view_11
#   linear_4 => convert_element_type_27
#   x => add_153
# Graph fragment:
#   %arg4_1 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mm_default_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg13_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_7 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf25 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=buf25]
#   %arg14_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %view_11 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_153 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %view_11), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_153, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_72 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_24, %getitem_7), kwargs = {})
#   %add_162 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_150 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %rsqrt_1), kwargs = {})
#   %mul_151 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %arg14_1), kwargs = {})
#   %add_163 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_151, %arg15_1), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_163, torch.bfloat16), kwargs = {})
#   return %getitem_7,%buf25,%convert_element_type_27
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/zv/czvqmm2tozis4izbcgxt5wkn55h65oojqbakbzcnrrpjq3ipqq46.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_4 => convert_element_type_26
# Graph fragment:
#   %arg16_1 : Tensor "f32[512, 256][256, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/qa/cqaynedsq5bfcr64t62pt7zx4ig7zgxb55ia7c3lju5mawdkasgi.py
# Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
# Source node to ATen node mapping:
#   layer_norm_1 => add_162, add_163, convert_element_type_24, mul_150, mul_151, rsqrt_1, sub_72, var_mean_1
#   linear_3 => add_tensor_23, convert_element_type_19, view_11
#   linear_4 => convert_element_type_26, convert_element_type_27, mm_default_22, permute_8, view_12
#   x => add_153
# Graph fragment:
#   %convert_element_type_27 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_27]
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_26]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %view_11 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_153 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %view_11), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_153, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_72 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_24, %getitem_7), kwargs = {})
#   %add_162 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_150 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %rsqrt_1), kwargs = {})
#   %mul_151 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %arg14_1), kwargs = {})
#   %add_163 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_151, %arg15_1), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_163, torch.bfloat16), kwargs = {})
#   %view_12 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_27, [%mul_25, 256]), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[512, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
#   %permute_8 : Tensor "bf16[256, 512][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_26, [1, 0]), kwargs = {})
#   %mm_default_22 : Tensor "bf16[s27*s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_12, %permute_8), kwargs = {})
#   return %mm_default_22
triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7(arg_A, arg_B, out_ptr0, ks0, ks1):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0*ks1
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/jq/cjq4tcf7623j2gso2eerq4lqkangvdec3gvy665imibmdwk4x5r2.py
# Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_190, convert_element_type_31, convert_element_type_32, erf, mul_176, mul_177, mul_178
#   linear_4 => add_tensor_22, convert_element_type_25, view_13
# Graph fragment:
#   %mm_default_22 : Tensor "bf16[s27*s77, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_22]
#   %arg17_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convert_element_type_25 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %add_tensor_22 : Tensor "bf16[s27*s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %convert_element_type_25), kwargs = {})
#   %view_13 : Tensor "bf16[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [%arg2_1, %arg3_1, 512]), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_176 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.5), kwargs = {})
#   %mul_177 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_177,), kwargs = {})
#   %add_190 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_178 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_176, %add_190), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_178, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32
triton_poi_fused__to_copy_addmm_gelu_view_8 = async_compile.triton('triton_poi_fused__to_copy_addmm_gelu_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_gelu_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_gelu_view_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/hz/chzgziitg5qhsxvxltgthqteaotbkllp5v6abvlzy5eifup53cgn.py
# Topologically Sorted Source Nodes: [linear_4, gelu, linear_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   gelu => add_190, convert_element_type_31, convert_element_type_32, erf, mul_176, mul_177, mul_178
#   linear_4 => add_tensor_22, convert_element_type_25, view_13
#   linear_5 => convert_element_type_34, mm_default_21, permute_9, view_14
# Graph fragment:
#   %convert_element_type_32 : Tensor "bf16[s77, s27, 512][512*s27, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_32]
#   %convert_element_type_34 : Tensor "bf16[256, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_34]
#   %convert_element_type_25 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %add_tensor_22 : Tensor "bf16[s27*s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %convert_element_type_25), kwargs = {})
#   %view_13 : Tensor "bf16[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [%arg2_1, %arg3_1, 512]), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_176 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.5), kwargs = {})
#   %mul_177 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_177,), kwargs = {})
#   %add_190 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_178 : Tensor "f32[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_176, %add_190), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s77, s27, 512][512*s27, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_178, torch.bfloat16), kwargs = {})
#   %view_14 : Tensor "bf16[s27*s77, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_32, [%mul_25, 512]), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[512, 256][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_34, [1, 0]), kwargs = {})
#   %mm_default_21 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_14, %permute_9), kwargs = {})
#   return %mm_default_21
triton_tem_fused__to_copy_addmm_gelu_t_view_9 = async_compile.triton('triton_tem_fused__to_copy_addmm_gelu_t_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_gelu_t_view_9', 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_gelu_t_view_9(arg_A, arg_B, out_ptr0, ks0, ks1):
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

    M = ks0*ks1
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/cg/ccgr36ck7mh4cce7spota2fqsgntyvqqdfeaipiylbhlf6did7mp.py
# Topologically Sorted Source Nodes: [linear_3, x, linear_5, x_1, layer_norm_2, linear_6, linear_7, linear_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_2 => add_214, add_215, convert_element_type_38, mul_206, mul_207, rsqrt_2, sub_95, var_mean_2
#   linear_3 => add_tensor_23, convert_element_type_19, view_11
#   linear_5 => add_tensor_21, convert_element_type_33, view_15
#   linear_6 => convert_element_type_41
#   linear_7 => convert_element_type_47
#   linear_8 => convert_element_type_53
#   x => add_153
#   x_1 => add_205
# Graph fragment:
#   %arg4_1 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mm_default_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg13_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %mm_default_21 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_21]
#   %arg19_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %add_205 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_205]
#   %getitem_9 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf35 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=buf35]
#   %arg20_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_215 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_215]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %add_tensor_23 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %convert_element_type_19), kwargs = {})
#   %view_11 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_153 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %view_11), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %add_tensor_21 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_21, %convert_element_type_33), kwargs = {})
#   %view_15 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_21, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_205 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %view_15), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_205, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_38, %getitem_9), kwargs = {})
#   %add_214 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_214,), kwargs = {})
#   %mul_206 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_2), kwargs = {})
#   %mul_207 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %arg20_1), kwargs = {})
#   %add_215 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %arg21_1), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_215, torch.bfloat16), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_215, torch.bfloat16), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_215, torch.bfloat16), kwargs = {})
#   return %add_205,%getitem_9,%buf35,%add_215,%convert_element_type_41,%convert_element_type_47,%convert_element_type_53
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/su/csuvvutlorqmpa7h6xbn46iy5k6kmohg7wk6oupetopq5gep7ch2.py
# Topologically Sorted Source Nodes: [linear_9, x_2, linear_11, x_3, layer_norm_4, linear_12, linear_13, linear_14], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_4 => add_424, add_425, convert_element_type_76, mul_405, mul_406, rsqrt_4, sub_188, var_mean_4
#   linear_11 => add_tensor_18, convert_element_type_71, view_31
#   linear_12 => convert_element_type_79
#   linear_13 => convert_element_type_85
#   linear_14 => convert_element_type_91
#   linear_9 => add_tensor_20, convert_element_type_57, view_27
#   x_2 => add_363
#   x_3 => add_415
# Graph fragment:
#   %add_205 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_205]
#   %mm_default_20 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_20]
#   %arg29_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %mm_default_18 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_18]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %add_415 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_415]
#   %getitem_17 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf69 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=buf69]
#   %arg36_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %arg37_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %add_425 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_425]
#   %convert_element_type_57 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %add_tensor_20 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %convert_element_type_57), kwargs = {})
#   %view_27 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_363 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_205, %view_27), kwargs = {})
#   %convert_element_type_71 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg35_1, torch.bfloat16), kwargs = {})
#   %add_tensor_18 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %convert_element_type_71), kwargs = {})
#   %view_31 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_18, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_415 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_363, %view_31), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_415, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_76, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_188 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_76, %getitem_17), kwargs = {})
#   %add_424 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_424,), kwargs = {})
#   %mul_405 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_188, %rsqrt_4), kwargs = {})
#   %mul_406 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_405, %arg36_1), kwargs = {})
#   %add_425 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_406, %arg37_1), kwargs = {})
#   %convert_element_type_79 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_425, torch.bfloat16), kwargs = {})
#   %convert_element_type_85 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_425, torch.bfloat16), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_425, torch.bfloat16), kwargs = {})
#   return %add_415,%getitem_17,%buf69,%add_425,%convert_element_type_79,%convert_element_type_85,%convert_element_type_91
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/bv/cbvjn4fjusrcfkxseygsnv37xk6m3a23vvzteb36tf2zdjfskfa4.py
# Topologically Sorted Source Nodes: [linear_36, q_6, transpose_24, linear_37, k_6, transpose_25, linear_38, v_6, transpose_26, getitem_27, attn_12, linear_42, q_7, transpose_28, linear_43, k_7, transpose_29, linear_44, v_7, transpose_30, getitem_31, attn_14], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_12 => _scaled_dot_product_efficient_attention_6, constant_pad_nd_6, expand_6, full_default_12, full_default_13, slice_21, where_6
#   attn_14 => _scaled_dot_product_efficient_attention_7, constant_pad_nd_7, expand_7, full_default_14, full_default_15, slice_24, where_7
#   getitem_27 => unsqueeze_12, unsqueeze_13
#   getitem_31 => unsqueeze_14, unsqueeze_15
#   k_6 => view_101
#   k_7 => view_117
#   linear_36 => view_97
#   linear_37 => view_100
#   linear_38 => view_103
#   linear_42 => view_113
#   linear_43 => view_116
#   linear_44 => view_119
#   q_6 => view_98
#   q_7 => view_114
#   transpose_24 => permute_63
#   transpose_25 => permute_64
#   transpose_26 => permute_65
#   transpose_28 => permute_73
#   transpose_29 => permute_74
#   transpose_30 => permute_75
#   v_6 => view_104
#   v_7 => view_120
# Graph fragment:
#   %arg5_1 : Tensor "b8[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %view_97 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_36, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_98 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_97, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_63 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_98, [0, 2, 1, 3]), kwargs = {})
#   %view_100 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_37, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_101 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_100, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_64 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_101, [0, 2, 1, 3]), kwargs = {})
#   %view_103 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_38, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_104 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_103, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_65 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_104, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_12 : Tensor "b8[s77, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %unsqueeze_13 : Tensor "b8[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 2), kwargs = {})
#   %full_default_13 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_12 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : Tensor "bf16[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_13, %full_default_13, %full_default_12), kwargs = {})
#   %constant_pad_nd_6 : Tensor "bf16[s77, 1, 1, s27 - (Mod(s27, 8)) + 8][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where_6, [0, %sub_49], 0.0), kwargs = {})
#   %slice_21 : Tensor "bf16[s77, 1, 1, s27][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_6, -1, 0, %arg3_1), kwargs = {})
#   %expand_6 : Tensor "bf16[s77, 8, s27, s27][Max(1, s27 - (Mod(s27, 8)) + 8), 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_21, [%arg2_1, 8, %arg3_1, %arg3_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_6 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_63, %permute_64, %permute_65, %expand_6, False), kwargs = {})
#   %view_113 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_42, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_114 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_113, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_73 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_114, [0, 2, 1, 3]), kwargs = {})
#   %view_116 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_43, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_117 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_116, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_74 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_117, [0, 2, 1, 3]), kwargs = {})
#   %view_119 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %view_120 : Tensor "bf16[s77, s27, 8, 32][256*s27, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_119, [%arg2_1, %arg3_1, 8, 32]), kwargs = {})
#   %permute_75 : Tensor "bf16[s77, 8, s27, 32][256*s27, 32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_14 : Tensor "b8[s77, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %unsqueeze_15 : Tensor "b8[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 2), kwargs = {})
#   %full_default_15 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_14 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_7 : Tensor "bf16[s77, 1, 1, s27][s27, s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_15, %full_default_15, %full_default_14), kwargs = {})
#   %constant_pad_nd_7 : Tensor "bf16[s77, 1, 1, s27 - (Mod(s27, 8)) + 8][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where_7, [0, %sub_49], 0.0), kwargs = {})
#   %slice_24 : Tensor "bf16[s77, 1, 1, s27][Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_7, -1, 0, %arg3_1), kwargs = {})
#   %expand_7 : Tensor "bf16[s77, 8, s27, s27][Max(1, s27 - (Mod(s27, 8)) + 8), 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_24, [%arg2_1, 8, %arg3_1, %arg3_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_7 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_73, %permute_74, %permute_75, %expand_7, False), kwargs = {})
#   return %buf220,%buf254
triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x2 = xindex // ks1
    x4 = xindex // ks0
    tmp0 = x0
    tmp1 = ks0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + ks0*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = 0.0
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tl.store(out_ptr0 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
    tl.store(out_ptr1 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/kg-v3/runs/gemm-limits-2026-09-29/inductor_cache/trunk_padded/lk/clk7tuwh5ai5hph7kjsqercercmjneu7oxl3ufliil237kekxlnm.py
# Topologically Sorted Source Nodes: [linear_45, x_14, linear_47, x_15, layer_norm_16], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_16 => add_1684, add_1685, convert_element_type_304, mul_1599, mul_1600, rsqrt_16, sub_746, var_mean_16
#   linear_45 => add_tensor_2, convert_element_type_285, view_123
#   linear_47 => add_tensor, convert_element_type_299, view_127
#   x_14 => add_1623
#   x_15 => add_1675
# Graph fragment:
#   %add_1465 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=add_1465]
#   %mm_default_2 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %arg125_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %mm_default : Tensor "bf16[s27*s77, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg131_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg131_1]
#   %convert_element_type_304 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_304]
#   %getitem_65 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=getitem_65]
#   %buf273 : Tensor "f32[s77, s27, 1][s27, 1, s27*s77]cuda:0" = PlaceHolder[target=buf273]
#   %arg132_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %arg133_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg133_1]
#   %convert_element_type_285 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   %add_tensor_2 : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %convert_element_type_285), kwargs = {})
#   %view_123 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_1623 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1465, %view_123), kwargs = {})
#   %convert_element_type_299 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg131_1, torch.bfloat16), kwargs = {})
#   %add_tensor : Tensor "bf16[s27*s77, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %convert_element_type_299), kwargs = {})
#   %view_127 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [%arg2_1, %arg3_1, 256]), kwargs = {})
#   %add_1675 : Tensor "bf16[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1623, %view_127), kwargs = {})
#   %convert_element_type_304 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1675, torch.float32), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_304, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_746 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_304, %getitem_65), kwargs = {})
#   %add_1684 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_64, 1e-05), kwargs = {})
#   %rsqrt_16 : Tensor "f32[s77, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1684,), kwargs = {})
#   %mul_1599 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_746, %rsqrt_16), kwargs = {})
#   %mul_1600 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1599, %arg132_1), kwargs = {})
#   %add_1685 : Tensor "f32[s77, s27, 256][256*s27, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1600, %arg133_1), kwargs = {})
#   return %convert_element_type_304,%getitem_65,%buf273,%add_1685
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '505DA61E28988D8397C94914DDA79AD82A0A5EA43BDB30419B2ED53AB792AAAA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1 = args
        args.clear()
        s77 = arg2_1
        s27 = arg3_1
        assert_size_stride(arg0_1, (256, ), (1, ))
        assert_size_stride(arg1_1, (256, ), (1, ))
        assert_size_stride(arg4_1, (s77, s27, 256), (256*s27, 256, 1))
        assert_size_stride(arg5_1, (s77, s27), (s27, 1))
        assert_size_stride(arg6_1, (256, 256), (256, 1))
        assert_size_stride(arg7_1, (256, ), (1, ))
        assert_size_stride(arg8_1, (256, 256), (256, 1))
        assert_size_stride(arg9_1, (256, ), (1, ))
        assert_size_stride(arg10_1, (256, 256), (256, 1))
        assert_size_stride(arg11_1, (256, ), (1, ))
        assert_size_stride(arg12_1, (256, 256), (256, 1))
        assert_size_stride(arg13_1, (256, ), (1, ))
        assert_size_stride(arg14_1, (256, ), (1, ))
        assert_size_stride(arg15_1, (256, ), (1, ))
        assert_size_stride(arg16_1, (512, 256), (256, 1))
        assert_size_stride(arg17_1, (512, ), (1, ))
        assert_size_stride(arg18_1, (256, 512), (512, 1))
        assert_size_stride(arg19_1, (256, ), (1, ))
        assert_size_stride(arg20_1, (256, ), (1, ))
        assert_size_stride(arg21_1, (256, ), (1, ))
        assert_size_stride(arg22_1, (256, 256), (256, 1))
        assert_size_stride(arg23_1, (256, ), (1, ))
        assert_size_stride(arg24_1, (256, 256), (256, 1))
        assert_size_stride(arg25_1, (256, ), (1, ))
        assert_size_stride(arg26_1, (256, 256), (256, 1))
        assert_size_stride(arg27_1, (256, ), (1, ))
        assert_size_stride(arg28_1, (256, 256), (256, 1))
        assert_size_stride(arg29_1, (256, ), (1, ))
        assert_size_stride(arg30_1, (256, ), (1, ))
        assert_size_stride(arg31_1, (256, ), (1, ))
        assert_size_stride(arg32_1, (512, 256), (256, 1))
        assert_size_stride(arg33_1, (512, ), (1, ))
        assert_size_stride(arg34_1, (256, 512), (512, 1))
        assert_size_stride(arg35_1, (256, ), (1, ))
        assert_size_stride(arg36_1, (256, ), (1, ))
        assert_size_stride(arg37_1, (256, ), (1, ))
        assert_size_stride(arg38_1, (256, 256), (256, 1))
        assert_size_stride(arg39_1, (256, ), (1, ))
        assert_size_stride(arg40_1, (256, 256), (256, 1))
        assert_size_stride(arg41_1, (256, ), (1, ))
        assert_size_stride(arg42_1, (256, 256), (256, 1))
        assert_size_stride(arg43_1, (256, ), (1, ))
        assert_size_stride(arg44_1, (256, 256), (256, 1))
        assert_size_stride(arg45_1, (256, ), (1, ))
        assert_size_stride(arg46_1, (256, ), (1, ))
        assert_size_stride(arg47_1, (256, ), (1, ))
        assert_size_stride(arg48_1, (512, 256), (256, 1))
        assert_size_stride(arg49_1, (512, ), (1, ))
        assert_size_stride(arg50_1, (256, 512), (512, 1))
        assert_size_stride(arg51_1, (256, ), (1, ))
        assert_size_stride(arg52_1, (256, ), (1, ))
        assert_size_stride(arg53_1, (256, ), (1, ))
        assert_size_stride(arg54_1, (256, 256), (256, 1))
        assert_size_stride(arg55_1, (256, ), (1, ))
        assert_size_stride(arg56_1, (256, 256), (256, 1))
        assert_size_stride(arg57_1, (256, ), (1, ))
        assert_size_stride(arg58_1, (256, 256), (256, 1))
        assert_size_stride(arg59_1, (256, ), (1, ))
        assert_size_stride(arg60_1, (256, 256), (256, 1))
        assert_size_stride(arg61_1, (256, ), (1, ))
        assert_size_stride(arg62_1, (256, ), (1, ))
        assert_size_stride(arg63_1, (256, ), (1, ))
        assert_size_stride(arg64_1, (512, 256), (256, 1))
        assert_size_stride(arg65_1, (512, ), (1, ))
        assert_size_stride(arg66_1, (256, 512), (512, 1))
        assert_size_stride(arg67_1, (256, ), (1, ))
        assert_size_stride(arg68_1, (256, ), (1, ))
        assert_size_stride(arg69_1, (256, ), (1, ))
        assert_size_stride(arg70_1, (256, 256), (256, 1))
        assert_size_stride(arg71_1, (256, ), (1, ))
        assert_size_stride(arg72_1, (256, 256), (256, 1))
        assert_size_stride(arg73_1, (256, ), (1, ))
        assert_size_stride(arg74_1, (256, 256), (256, 1))
        assert_size_stride(arg75_1, (256, ), (1, ))
        assert_size_stride(arg76_1, (256, 256), (256, 1))
        assert_size_stride(arg77_1, (256, ), (1, ))
        assert_size_stride(arg78_1, (256, ), (1, ))
        assert_size_stride(arg79_1, (256, ), (1, ))
        assert_size_stride(arg80_1, (512, 256), (256, 1))
        assert_size_stride(arg81_1, (512, ), (1, ))
        assert_size_stride(arg82_1, (256, 512), (512, 1))
        assert_size_stride(arg83_1, (256, ), (1, ))
        assert_size_stride(arg84_1, (256, ), (1, ))
        assert_size_stride(arg85_1, (256, ), (1, ))
        assert_size_stride(arg86_1, (256, 256), (256, 1))
        assert_size_stride(arg87_1, (256, ), (1, ))
        assert_size_stride(arg88_1, (256, 256), (256, 1))
        assert_size_stride(arg89_1, (256, ), (1, ))
        assert_size_stride(arg90_1, (256, 256), (256, 1))
        assert_size_stride(arg91_1, (256, ), (1, ))
        assert_size_stride(arg92_1, (256, 256), (256, 1))
        assert_size_stride(arg93_1, (256, ), (1, ))
        assert_size_stride(arg94_1, (256, ), (1, ))
        assert_size_stride(arg95_1, (256, ), (1, ))
        assert_size_stride(arg96_1, (512, 256), (256, 1))
        assert_size_stride(arg97_1, (512, ), (1, ))
        assert_size_stride(arg98_1, (256, 512), (512, 1))
        assert_size_stride(arg99_1, (256, ), (1, ))
        assert_size_stride(arg100_1, (256, ), (1, ))
        assert_size_stride(arg101_1, (256, ), (1, ))
        assert_size_stride(arg102_1, (256, 256), (256, 1))
        assert_size_stride(arg103_1, (256, ), (1, ))
        assert_size_stride(arg104_1, (256, 256), (256, 1))
        assert_size_stride(arg105_1, (256, ), (1, ))
        assert_size_stride(arg106_1, (256, 256), (256, 1))
        assert_size_stride(arg107_1, (256, ), (1, ))
        assert_size_stride(arg108_1, (256, 256), (256, 1))
        assert_size_stride(arg109_1, (256, ), (1, ))
        assert_size_stride(arg110_1, (256, ), (1, ))
        assert_size_stride(arg111_1, (256, ), (1, ))
        assert_size_stride(arg112_1, (512, 256), (256, 1))
        assert_size_stride(arg113_1, (512, ), (1, ))
        assert_size_stride(arg114_1, (256, 512), (512, 1))
        assert_size_stride(arg115_1, (256, ), (1, ))
        assert_size_stride(arg116_1, (256, ), (1, ))
        assert_size_stride(arg117_1, (256, ), (1, ))
        assert_size_stride(arg118_1, (256, 256), (256, 1))
        assert_size_stride(arg119_1, (256, ), (1, ))
        assert_size_stride(arg120_1, (256, 256), (256, 1))
        assert_size_stride(arg121_1, (256, ), (1, ))
        assert_size_stride(arg122_1, (256, 256), (256, 1))
        assert_size_stride(arg123_1, (256, ), (1, ))
        assert_size_stride(arg124_1, (256, 256), (256, 1))
        assert_size_stride(arg125_1, (256, ), (1, ))
        assert_size_stride(arg126_1, (256, ), (1, ))
        assert_size_stride(arg127_1, (256, ), (1, ))
        assert_size_stride(arg128_1, (512, 256), (256, 1))
        assert_size_stride(arg129_1, (512, ), (1, ))
        assert_size_stride(arg130_1, (256, 512), (512, 1))
        assert_size_stride(arg131_1, (256, ), (1, ))
        assert_size_stride(arg132_1, (256, ), (1, ))
        assert_size_stride(arg133_1, (256, ), (1, ))
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        _xnumel = 512*s27*s77
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf4 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [layer_norm, linear, linear_1, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm]
            triton_per_fused__to_copy_native_layer_norm_0_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_0.run(arg4_1, arg0_1, arg1_1, buf4, buf8, buf12, triton_per_fused__to_copy_native_layer_norm_0_xnumel, 256, stream=stream0)
            del arg0_1
            del arg1_1
            buf5 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg6_1, buf5, 65536, stream=stream0)
            del arg6_1
            buf6 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg7_1, buf6, 256, stream=stream0)
            del arg7_1
            buf7 = empty_strided_cuda((s27*s77, 256), (256, 1), torch.bfloat16)
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf6, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf4, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf5, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf7)
            buf9 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg8_1, buf9, 65536, stream=stream0)
            del arg8_1
            buf10 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg9_1, buf10, 256, stream=stream0)
            del arg9_1
            buf11 = reinterpret_tensor(buf4, (s27*s77, 256), (256, 1), 0); del buf4  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf10, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf8, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf9, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf11)
            buf13 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg10_1, buf13, 65536, stream=stream0)
            del arg10_1
            buf14 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg11_1, buf14, 256, stream=stream0)
            del arg11_1
            buf15 = reinterpret_tensor(buf8, (s27*s77, 256), (256, 1), 0); del buf8  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf14, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf12, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf13, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf15)
            del buf12
            ps0 = s27*s27
            buf16 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            buf50 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            buf84 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, q, transpose, linear_1, k, transpose_1, linear_2, v, transpose_2, getitem_3, attn, linear_6, q_1, transpose_4, linear_7, k_1, transpose_5, linear_8, v_1, transpose_6, getitem_7, attn_2, linear_12, q_2, transpose_8, linear_13, k_2, transpose_9, linear_14, v_2, transpose_10, getitem_11, attn_4], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3_xnumel = s77*s27*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3.run(arg5_1, buf16, buf50, buf84, s27, ps0, triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [linear, q, transpose, linear_1, k, transpose_1, linear_2, v, transpose_2, getitem_3, attn], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf7, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf11, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf15, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf16, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf16
            buf18 = buf17[0]
            assert_size_stride(buf18, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf17
            buf22 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg12_1, buf22, 65536, stream=stream0)
            del arg12_1
            buf23 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [attn_1, reshape, linear_3], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf18, buf22, buf23, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf27 = reinterpret_tensor(buf18, (s77, s27, 256), (256*s27, 256, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(arg4_1, buf23, arg13_1, arg14_1, arg15_1, buf27, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg14_1
            del arg15_1
            buf28 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg16_1, buf28, 131072, stream=stream0)
            del arg16_1
            buf29 = empty_strided_cuda((s27*s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, x, layer_norm_1, linear_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf27, buf28, buf29, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf30 = reinterpret_tensor(buf29, (s77, s27, 512), (512*s27, 512, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf30, arg17_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg17_1
            buf31 = reinterpret_tensor(buf28, (256, 512), (512, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg18_1, buf31, 131072, stream=stream0)
            del arg18_1
            buf32 = reinterpret_tensor(buf27, (s27*s77, 256), (256, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu, linear_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf30, buf31, buf32, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf30
            buf33 = reinterpret_tensor(buf23, (s77, s27, 256), (256*s27, 256, 1), 0); del buf23  # reuse
            buf38 = reinterpret_tensor(buf15, (s77, s27, 256), (256*s27, 256, 1), 0); del buf15  # reuse
            buf42 = reinterpret_tensor(buf11, (s77, s27, 256), (256*s27, 256, 1), 0); del buf11  # reuse
            buf46 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_3, x, linear_5, x_1, layer_norm_2, linear_6, linear_7, linear_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10.run(buf33, arg4_1, arg13_1, buf32, arg19_1, arg20_1, arg21_1, buf38, buf42, buf46, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_10_xnumel, 256, stream=stream0)
            del arg13_1
            del arg19_1
            del arg20_1
            del arg21_1
            del arg4_1
            buf39 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg22_1, buf39, 65536, stream=stream0)
            del arg22_1
            buf40 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg23_1, buf40, 256, stream=stream0)
            del arg23_1
            buf41 = buf32; del buf32  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf40, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf38, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf39, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf41)
            buf43 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg24_1, buf43, 65536, stream=stream0)
            del arg24_1
            buf44 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg25_1, buf44, 256, stream=stream0)
            del arg25_1
            buf45 = reinterpret_tensor(buf38, (s27*s77, 256), (256, 1), 0); del buf38  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf44, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf42, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf43, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf45)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg26_1, buf47, 65536, stream=stream0)
            del arg26_1
            buf48 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg27_1, buf48, 256, stream=stream0)
            del arg27_1
            buf49 = reinterpret_tensor(buf42, (s27*s77, 256), (256, 1), 0); del buf42  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf48, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf46, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf47, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf49)
            # Topologically Sorted Source Nodes: [linear_6, q_1, transpose_4, linear_7, k_1, transpose_5, linear_8, v_1, transpose_6, getitem_7, attn_2], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf41, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf45, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf49, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf50, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf50
            buf52 = buf51[0]
            assert_size_stride(buf52, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf52, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf51
            buf56 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg28_1, buf56, 65536, stream=stream0)
            del arg28_1
            buf57 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [attn_3, reshape_1, linear_9], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf52, buf56, buf57, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf61 = reinterpret_tensor(buf52, (s77, s27, 256), (256*s27, 256, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [linear_9, x_2, layer_norm_3, linear_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf33, buf57, arg29_1, arg30_1, arg31_1, buf61, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg30_1
            del arg31_1
            buf62 = reinterpret_tensor(buf31, (512, 256), (256, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg32_1, buf62, 131072, stream=stream0)
            del arg32_1
            buf63 = empty_strided_cuda((s27*s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, x_2, layer_norm_3, linear_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf61, buf62, buf63, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf64 = reinterpret_tensor(buf63, (s77, s27, 512), (512*s27, 512, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_10, gelu_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf64, arg33_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg33_1
            buf65 = reinterpret_tensor(buf62, (256, 512), (512, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg34_1, buf65, 131072, stream=stream0)
            del arg34_1
            buf66 = reinterpret_tensor(buf61, (s27*s77, 256), (256, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [linear_10, gelu_1, linear_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf64, buf65, buf66, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf67 = buf33; del buf33  # reuse
            buf72 = reinterpret_tensor(buf45, (s77, s27, 256), (256*s27, 256, 1), 0); del buf45  # reuse
            buf76 = reinterpret_tensor(buf41, (s77, s27, 256), (256*s27, 256, 1), 0); del buf41  # reuse
            buf80 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [linear_9, x_2, linear_11, x_3, layer_norm_4, linear_12, linear_13, linear_14], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf67, buf57, arg29_1, buf66, arg35_1, arg36_1, arg37_1, buf72, buf76, buf80, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg29_1
            del arg35_1
            del arg36_1
            del arg37_1
            del buf57
            buf73 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg38_1, buf73, 65536, stream=stream0)
            del arg38_1
            buf74 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg39_1, buf74, 256, stream=stream0)
            del arg39_1
            buf75 = buf66; del buf66  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf74, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf72, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf73, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf75)
            buf77 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg40_1, buf77, 65536, stream=stream0)
            del arg40_1
            buf78 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg41_1, buf78, 256, stream=stream0)
            del arg41_1
            buf79 = reinterpret_tensor(buf72, (s27*s77, 256), (256, 1), 0); del buf72  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf78, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf76, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf77, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf79)
            buf81 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg42_1, buf81, 65536, stream=stream0)
            del arg42_1
            buf82 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg43_1, buf82, 256, stream=stream0)
            del arg43_1
            buf83 = reinterpret_tensor(buf76, (s27*s77, 256), (256, 1), 0); del buf76  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf82, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf80, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf81, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf83)
            # Topologically Sorted Source Nodes: [linear_12, q_2, transpose_8, linear_13, k_2, transpose_9, linear_14, v_2, transpose_10, getitem_11, attn_4], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf85 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf75, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf79, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf83, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf84, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            buf86 = buf85[0]
            assert_size_stride(buf86, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf86, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf85
            buf90 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg44_1, buf90, 65536, stream=stream0)
            del arg44_1
            buf91 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [attn_5, reshape_2, linear_15], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf86, buf90, buf91, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf95 = reinterpret_tensor(buf86, (s77, s27, 256), (256*s27, 256, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [linear_15, x_4, layer_norm_5, linear_16], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf67, buf91, arg45_1, arg46_1, arg47_1, buf95, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg46_1
            del arg47_1
            buf96 = reinterpret_tensor(buf65, (512, 256), (256, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg48_1, buf96, 131072, stream=stream0)
            del arg48_1
            buf97 = reinterpret_tensor(buf64, (s27*s77, 512), (512, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_15, x_4, layer_norm_5, linear_16], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf95, buf96, buf97, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf98 = reinterpret_tensor(buf97, (s77, s27, 512), (512*s27, 512, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [linear_16, gelu_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf98, arg49_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg49_1
            buf99 = reinterpret_tensor(buf96, (256, 512), (512, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg50_1, buf99, 131072, stream=stream0)
            del arg50_1
            buf100 = reinterpret_tensor(buf95, (s27*s77, 256), (256, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [linear_16, gelu_2, linear_17], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf98, buf99, buf100, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf98
            buf101 = buf67; del buf67  # reuse
            buf106 = reinterpret_tensor(buf79, (s77, s27, 256), (256*s27, 256, 1), 0); del buf79  # reuse
            buf110 = reinterpret_tensor(buf75, (s77, s27, 256), (256*s27, 256, 1), 0); del buf75  # reuse
            buf114 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [linear_15, x_4, linear_17, x_5, layer_norm_6, linear_18, linear_19, linear_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf101, buf91, arg45_1, buf100, arg51_1, arg52_1, arg53_1, buf106, buf110, buf114, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg45_1
            del arg51_1
            del arg52_1
            del arg53_1
            del buf100
            buf107 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg54_1, buf107, 65536, stream=stream0)
            del arg54_1
            buf108 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg55_1, buf108, 256, stream=stream0)
            del arg55_1
            buf109 = buf91; del buf91  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf108, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf106, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf107, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf109)
            buf111 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg56_1, buf111, 65536, stream=stream0)
            del arg56_1
            buf112 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg57_1, buf112, 256, stream=stream0)
            del arg57_1
            buf113 = reinterpret_tensor(buf106, (s27*s77, 256), (256, 1), 0); del buf106  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf112, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf110, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf111, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf113)
            buf115 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg58_1, buf115, 65536, stream=stream0)
            del arg58_1
            buf116 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg59_1, buf116, 256, stream=stream0)
            del arg59_1
            buf117 = reinterpret_tensor(buf110, (s27*s77, 256), (256, 1), 0); del buf110  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf116, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf114, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf115, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf117)
            del buf114
            del buf116
            buf118 = buf84; del buf84  # reuse
            buf152 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            buf186 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18, q_3, transpose_12, linear_19, k_3, transpose_13, linear_20, v_3, transpose_14, getitem_15, attn_6, linear_24, q_4, transpose_16, linear_25, k_4, transpose_17, linear_26, v_4, transpose_18, getitem_19, attn_8, linear_30, q_5, transpose_20, linear_31, k_5, transpose_21, linear_32, v_5, transpose_22, getitem_23, attn_10], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3_xnumel = s77*s27*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3.run(arg5_1, buf118, buf152, buf186, s27, ps0, triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_3_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [linear_18, q_3, transpose_12, linear_19, k_3, transpose_13, linear_20, v_3, transpose_14, getitem_15, attn_6], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf109, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf113, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf117, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf118, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf109
            del buf118
            buf120 = buf119[0]
            assert_size_stride(buf120, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf120, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf119
            buf124 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg60_1, buf124, 65536, stream=stream0)
            del arg60_1
            buf125 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [attn_7, reshape_3, linear_21], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf120, buf124, buf125, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf124
            buf129 = reinterpret_tensor(buf120, (s77, s27, 256), (256*s27, 256, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [linear_21, x_6, layer_norm_7, linear_22], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf101, buf125, arg61_1, arg62_1, arg63_1, buf129, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg62_1
            del arg63_1
            buf130 = reinterpret_tensor(buf99, (512, 256), (256, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg64_1, buf130, 131072, stream=stream0)
            del arg64_1
            buf131 = empty_strided_cuda((s27*s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_21, x_6, layer_norm_7, linear_22], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf129, buf130, buf131, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf132 = reinterpret_tensor(buf131, (s77, s27, 512), (512*s27, 512, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_22, gelu_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf132, arg65_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg65_1
            buf133 = reinterpret_tensor(buf130, (256, 512), (512, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg66_1, buf133, 131072, stream=stream0)
            del arg66_1
            buf134 = reinterpret_tensor(buf129, (s27*s77, 256), (256, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_22, gelu_3, linear_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf132, buf133, buf134, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf132
            del buf133
            buf135 = buf101; del buf101  # reuse
            buf140 = reinterpret_tensor(buf113, (s77, s27, 256), (256*s27, 256, 1), 0); del buf113  # reuse
            buf144 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            buf148 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_21, x_6, linear_23, x_7, layer_norm_8, linear_24, linear_25, linear_26], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf135, buf125, arg61_1, buf134, arg67_1, arg68_1, arg69_1, buf140, buf144, buf148, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg61_1
            del arg67_1
            del arg68_1
            del arg69_1
            del buf125
            buf141 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg70_1, buf141, 65536, stream=stream0)
            del arg70_1
            buf142 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg71_1, buf142, 256, stream=stream0)
            del arg71_1
            buf143 = buf134; del buf134  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf142, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf140, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf141, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf143)
            buf145 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg72_1, buf145, 65536, stream=stream0)
            del arg72_1
            buf146 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg73_1, buf146, 256, stream=stream0)
            del arg73_1
            buf147 = reinterpret_tensor(buf140, (s27*s77, 256), (256, 1), 0); del buf140  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf146, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf144, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf145, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf147)
            buf149 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg74_1, buf149, 65536, stream=stream0)
            del arg74_1
            buf150 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg75_1, buf150, 256, stream=stream0)
            del arg75_1
            buf151 = reinterpret_tensor(buf144, (s27*s77, 256), (256, 1), 0); del buf144  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf150, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf148, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf149, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf151)
            # Topologically Sorted Source Nodes: [linear_24, q_4, transpose_16, linear_25, k_4, transpose_17, linear_26, v_4, transpose_18, getitem_19, attn_8], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf153 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf143, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf147, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf151, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf152, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf152
            buf154 = buf153[0]
            assert_size_stride(buf154, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf154, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf153
            buf158 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg76_1, buf158, 65536, stream=stream0)
            del arg76_1
            buf159 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [attn_9, reshape_4, linear_27], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf154, buf158, buf159, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf163 = reinterpret_tensor(buf154, (s77, s27, 256), (256*s27, 256, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [linear_27, x_8, layer_norm_9, linear_28], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf135, buf159, arg77_1, arg78_1, arg79_1, buf163, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg78_1
            del arg79_1
            buf164 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg80_1, buf164, 131072, stream=stream0)
            del arg80_1
            buf165 = empty_strided_cuda((s27*s77, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27, x_8, layer_norm_9, linear_28], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf163, buf164, buf165, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf166 = reinterpret_tensor(buf165, (s77, s27, 512), (512*s27, 512, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [linear_28, gelu_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf166, arg81_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg81_1
            buf167 = reinterpret_tensor(buf164, (256, 512), (512, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg82_1, buf167, 131072, stream=stream0)
            del arg82_1
            buf168 = reinterpret_tensor(buf163, (s27*s77, 256), (256, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [linear_28, gelu_4, linear_29], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf166, buf167, buf168, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf169 = buf135; del buf135  # reuse
            buf174 = reinterpret_tensor(buf147, (s77, s27, 256), (256*s27, 256, 1), 0); del buf147  # reuse
            buf178 = reinterpret_tensor(buf143, (s77, s27, 256), (256*s27, 256, 1), 0); del buf143  # reuse
            buf182 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [linear_27, x_8, linear_29, x_9, layer_norm_10, linear_30, linear_31, linear_32], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf169, buf159, arg77_1, buf168, arg83_1, arg84_1, arg85_1, buf174, buf178, buf182, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg77_1
            del arg83_1
            del arg84_1
            del arg85_1
            del buf159
            buf175 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg86_1, buf175, 65536, stream=stream0)
            del arg86_1
            buf176 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg87_1, buf176, 256, stream=stream0)
            del arg87_1
            buf177 = buf168; del buf168  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf176, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf174, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf175, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf177)
            buf179 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg88_1, buf179, 65536, stream=stream0)
            del arg88_1
            buf180 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg89_1, buf180, 256, stream=stream0)
            del arg89_1
            buf181 = reinterpret_tensor(buf174, (s27*s77, 256), (256, 1), 0); del buf174  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf180, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf178, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf179, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf181)
            buf183 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg90_1, buf183, 65536, stream=stream0)
            del arg90_1
            buf184 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg91_1, buf184, 256, stream=stream0)
            del arg91_1
            buf185 = reinterpret_tensor(buf178, (s27*s77, 256), (256, 1), 0); del buf178  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf184, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf182, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf183, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf185)
            # Topologically Sorted Source Nodes: [linear_30, q_5, transpose_20, linear_31, k_5, transpose_21, linear_32, v_5, transpose_22, getitem_23, attn_10], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf187 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf177, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf181, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf185, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf186, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            buf188 = buf187[0]
            assert_size_stride(buf188, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf188, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf187
            buf192 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg92_1, buf192, 65536, stream=stream0)
            del arg92_1
            buf193 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [attn_11, reshape_5, linear_33], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf188, buf192, buf193, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf197 = reinterpret_tensor(buf188, (s77, s27, 256), (256*s27, 256, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [linear_33, x_10, layer_norm_11, linear_34], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf169, buf193, arg93_1, arg94_1, arg95_1, buf197, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg94_1
            del arg95_1
            buf198 = reinterpret_tensor(buf167, (512, 256), (256, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg96_1, buf198, 131072, stream=stream0)
            del arg96_1
            buf199 = reinterpret_tensor(buf166, (s27*s77, 512), (512, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [linear_33, x_10, layer_norm_11, linear_34], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf197, buf198, buf199, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf200 = reinterpret_tensor(buf199, (s77, s27, 512), (512*s27, 512, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [linear_34, gelu_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf200, arg97_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg97_1
            buf201 = reinterpret_tensor(buf198, (256, 512), (512, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg98_1, buf201, 131072, stream=stream0)
            del arg98_1
            buf202 = reinterpret_tensor(buf197, (s27*s77, 256), (256, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [linear_34, gelu_5, linear_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf200, buf201, buf202, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf203 = buf169; del buf169  # reuse
            buf208 = reinterpret_tensor(buf181, (s77, s27, 256), (256*s27, 256, 1), 0); del buf181  # reuse
            buf212 = reinterpret_tensor(buf177, (s77, s27, 256), (256*s27, 256, 1), 0); del buf177  # reuse
            buf216 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [linear_33, x_10, linear_35, x_11, layer_norm_12, linear_36, linear_37, linear_38], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf203, buf193, arg93_1, buf202, arg99_1, arg100_1, arg101_1, buf208, buf212, buf216, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg100_1
            del arg101_1
            del arg93_1
            del arg99_1
            del buf193
            buf209 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg102_1, buf209, 65536, stream=stream0)
            del arg102_1
            buf210 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg103_1, buf210, 256, stream=stream0)
            del arg103_1
            buf211 = buf202; del buf202  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf210, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf208, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf209, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf211)
            buf213 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg104_1, buf213, 65536, stream=stream0)
            del arg104_1
            buf214 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg105_1, buf214, 256, stream=stream0)
            del arg105_1
            buf215 = reinterpret_tensor(buf208, (s27*s77, 256), (256, 1), 0); del buf208  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf214, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf212, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf213, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf215)
            buf217 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg106_1, buf217, 65536, stream=stream0)
            del arg106_1
            buf218 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg107_1, buf218, 256, stream=stream0)
            del arg107_1
            buf219 = reinterpret_tensor(buf212, (s27*s77, 256), (256, 1), 0); del buf212  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf218, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf216, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf217, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf219)
            buf220 = buf186; del buf186  # reuse
            buf254 = empty_strided_cuda((s77, 1, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36, q_6, transpose_24, linear_37, k_6, transpose_25, linear_38, v_6, transpose_26, getitem_27, attn_12, linear_42, q_7, transpose_28, linear_43, k_7, transpose_29, linear_44, v_7, transpose_30, getitem_31, attn_14], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12_xnumel = s77*s27*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12.run(arg5_1, buf220, buf254, s27, ps0, triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_expand_scalar_tensor_slice_transpose_unsqueeze_view_where_12_xnumel, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [linear_36, q_6, transpose_24, linear_37, k_6, transpose_25, linear_38, v_6, transpose_26, getitem_27, attn_12], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf221 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf211, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf215, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf219, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf220, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf220
            buf222 = buf221[0]
            assert_size_stride(buf222, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf222, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf221
            buf226 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg108_1, buf226, 65536, stream=stream0)
            del arg108_1
            buf227 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [attn_13, reshape_6, linear_39], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf222, buf226, buf227, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf231 = reinterpret_tensor(buf222, (s77, s27, 256), (256*s27, 256, 1), 0); del buf222  # reuse
            # Topologically Sorted Source Nodes: [linear_39, x_12, layer_norm_13, linear_40], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf203, buf227, arg109_1, arg110_1, arg111_1, buf231, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg110_1
            del arg111_1
            buf232 = reinterpret_tensor(buf201, (512, 256), (256, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg112_1, buf232, 131072, stream=stream0)
            del arg112_1
            buf233 = reinterpret_tensor(buf200, (s27*s77, 512), (512, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [linear_39, x_12, layer_norm_13, linear_40], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf231, buf232, buf233, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf234 = reinterpret_tensor(buf233, (s77, s27, 512), (512*s27, 512, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [linear_40, gelu_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf234, arg113_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg113_1
            buf235 = reinterpret_tensor(buf232, (256, 512), (512, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg114_1, buf235, 131072, stream=stream0)
            del arg114_1
            buf236 = reinterpret_tensor(buf231, (s27*s77, 256), (256, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [linear_40, gelu_6, linear_41], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf234, buf235, buf236, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            buf237 = buf203; del buf203  # reuse
            buf242 = reinterpret_tensor(buf215, (s77, s27, 256), (256*s27, 256, 1), 0); del buf215  # reuse
            buf246 = reinterpret_tensor(buf211, (s77, s27, 256), (256*s27, 256, 1), 0); del buf211  # reuse
            buf250 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [linear_39, x_12, linear_41, x_13, layer_norm_14, linear_42, linear_43, linear_44], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11.run(buf237, buf227, arg109_1, buf236, arg115_1, arg116_1, arg117_1, buf242, buf246, buf250, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_11_xnumel, 256, stream=stream0)
            del arg109_1
            del arg115_1
            del arg116_1
            del arg117_1
            del buf227
            buf243 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg118_1, buf243, 65536, stream=stream0)
            del arg118_1
            buf244 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg119_1, buf244, 256, stream=stream0)
            del arg119_1
            buf245 = buf236; del buf236  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf244, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf242, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf243, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf245)
            buf247 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg120_1, buf247, 65536, stream=stream0)
            del arg120_1
            buf248 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg121_1, buf248, 256, stream=stream0)
            del arg121_1
            buf249 = reinterpret_tensor(buf242, (s27*s77, 256), (256, 1), 0); del buf242  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf248, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf246, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf247, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf249)
            buf251 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg122_1, buf251, 65536, stream=stream0)
            del arg122_1
            buf252 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg123_1, buf252, 256, stream=stream0)
            del arg123_1
            buf253 = reinterpret_tensor(buf246, (s27*s77, 256), (256, 1), 0); del buf246  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            extern_kernels.bias_addmm(reinterpret_tensor(buf252, (s27*s77, 256), (0, 1), 0), reinterpret_tensor(buf250, (s27*s77, 256), (256, 1), 0), reinterpret_tensor(buf251, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf253)
            del buf250
            del buf252
            # Topologically Sorted Source Nodes: [linear_42, q_7, transpose_28, linear_43, k_7, transpose_29, linear_44, v_7, transpose_30, getitem_31, attn_14], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf255 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf245, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf249, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf253, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 0), reinterpret_tensor(buf254, (s77, 8, s27, s27), (8*s27*((7 + s27) // 8), 0, 8*((7 + s27) // 8), 1), 0), False)
            del buf245
            del buf249
            del buf254
            buf256 = buf255[0]
            assert_size_stride(buf256, (s77, 8, s27, 32), (256*s27, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf256, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf255
            buf260 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg124_1, buf260, 65536, stream=stream0)
            del arg124_1
            buf261 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [attn_15, reshape_7, linear_45], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_t_transpose_view_4.run(buf256, buf260, buf261, s27, s77, 4*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf260
            buf265 = reinterpret_tensor(buf256, (s77, s27, 256), (256*s27, 256, 1), 0); del buf256  # reuse
            # Topologically Sorted Source Nodes: [linear_45, x_14, layer_norm_15, linear_46], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5.run(buf237, buf261, arg125_1, arg126_1, arg127_1, buf265, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_5_xnumel, 256, stream=stream0)
            del arg126_1
            del arg127_1
            buf266 = reinterpret_tensor(buf235, (512, 256), (256, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg128_1, buf266, 131072, stream=stream0)
            del arg128_1
            buf267 = reinterpret_tensor(buf234, (s27*s77, 512), (512, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [linear_45, x_14, layer_norm_15, linear_46], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_native_layer_norm_t_view_7.run(buf265, buf266, buf267, s27, s77, 4*((127 + s27*s77) // 128), 1, 1, stream=stream0)
            buf268 = reinterpret_tensor(buf267, (s77, s27, 512), (512*s27, 512, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [linear_46, gelu_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
            triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel = 512*s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_gelu_view_8.run(buf268, arg129_1, triton_poi_fused__to_copy_addmm_gelu_view_8_xnumel, stream=stream0)
            del arg129_1
            buf269 = reinterpret_tensor(buf266, (256, 512), (512, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg130_1, buf269, 131072, stream=stream0)
            del arg130_1
            buf270 = reinterpret_tensor(buf265, (s27*s77, 256), (256, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [linear_46, gelu_7, linear_47], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_addmm_gelu_t_view_9.run(buf268, buf269, buf270, s27, s77, 2*((63 + s27*s77) // 64), 1, 1, stream=stream0)
            del buf268
            del buf269
            buf271 = empty_strided_cuda((s77, s27, 256), (256*s27, 256, 1), torch.float32)
            buf275 = buf271; del buf271  # reuse
            # Topologically Sorted Source Nodes: [linear_45, x_14, linear_47, x_15, layer_norm_16], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13.run(buf275, buf237, buf261, arg125_1, buf270, arg131_1, arg132_1, arg133_1, triton_per_fused__to_copy_add_addmm_native_layer_norm_view_13_xnumel, 256, stream=stream0)
            del arg125_1
            del arg131_1
            del arg132_1
            del arg133_1
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
    arg2_1 = 8
    arg3_1 = 709
    arg4_1 = rand_strided((8, 709, 256), (181504, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8, 709), (709, 1), device='cuda:0', dtype=torch.bool)
    arg6_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
