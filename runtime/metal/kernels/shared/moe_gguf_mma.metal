// GGUF MoE experts for Apple7/8 (M1/M2) on the simdgroup MMA tile (kernels/common/gguf_mma_tile.h): the bindings,
// grid and outputs of moe_expert_gguf_m* (kernels/shared/moe_gguf.metal), whose MPP matmul2d runs far below the MMA
// rate there. A tile runs the 8-, 16- or 32-row MMAs that hold its live rows.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

using namespace gguf_mma;

template <ushort Rows, GgufEpilogue Ep, uint Shared>
kernel void moe_expert_gguf_mma(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                                device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                                device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                                device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                                device const uint *slots [[buffer(11)]], constant MoeGgufExpertParams &p [[buffer(12)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment<Shared>(tile.expert, p, slots, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  const auto run = [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    accumulate_any<FM>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, stage + simd_group * kSimdgroupStage, tl,
                       thread_index, GGUF_STAGED_THREADS, simd_lane, 0, p.input_size / kStep, acc);
    elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[o]));
      output[o] = bfloat(v);
    });
  };
  if constexpr (Rows == 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 16) run(integral_constant<uint, 2>{});
  else run(integral_constant<uint, 4>{});
}
using MoeExpertGgufMmaKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                    device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                    device bfloat *, device bfloat *, device const uint *, constant MoeGgufExpertParams &,
                                    uint2, uint, uint);
template [[host_name("moe_expert_gguf_mma_m8_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone, 1>;
template [[host_name("moe_expert_gguf_mma_m8_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate, 1>;
template [[host_name("moe_expert_gguf_mma_m32_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone, 1>;
template [[host_name("moe_expert_gguf_mma_m32_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate, 1>;
template [[host_name("moe_expert_gguf_mma_m8_a_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone, 2>;
template [[host_name("moe_expert_gguf_mma_m8_g_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate, 2>;
template [[host_name("moe_expert_gguf_mma_m32_a_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone, 2>;
template [[host_name("moe_expert_gguf_mma_m32_g_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate, 2>;

// One simdgroup for gate and one for up, over the same 32 columns. Each runs
// the original MMA sequence; gate's finished stage holds its bf16 output.
// The up epilogue keeps both original bf16 rounding points.
template <ushort Rows, ushort Tables, uint Shared>
kernel void moe_expert_gguf_mma_gate_up(device bfloat *input [[buffer(0)]],
                                        device const MoeTileDescriptor *tiles [[buffer(1)]],
                                        device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]],
                                        device uchar *gw1 [[buffer(4)]], device uchar *gmeta [[buffer(5)]],
                                        device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
                                        device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]],
                                        device uchar *uw1 [[buffer(10)]], device uchar *umeta [[buffer(11)]],
                                        device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
                                        device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]],
                                        device const uint *slots [[buffer(16)]],
                                        constant MoeGgufGateUpParams &p [[buffer(17)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint simd_lane [[thread_index_in_simdgroup]],
                                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  static_assert(Rows * kColumns <= kSimdgroupStage, "gate fits in its finished stage");
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[Tables * kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                   p.gate_routed_format, p.gate_shared_format, slots,
                                                   gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                   p.up_routed_format, p.up_shared_format, slots,
                                                   uw0, uw1, umeta, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  bool tables = false;
  quant_format_switch(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    quant_format_switch(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index, GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool up = simd_group == 1;
  const MoeGgufSegment s = up ? us : gs;
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const uint origin = group.x * kColumns;
  const ulong out = ulong(group.y) * Rows * p.output_size + origin;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  const auto run = [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    quant_format_switch(s.format, [&](auto format) {
      accumulate<decltype(format), FM>(x, s.w0, s.w1, s.meta, p.input_size, origin,
                                       stage + simd_group * kSimdgroupStage, table, simd_lane,
                                       0, p.input_size / kStep, acc);
    });
    if (!up)
      elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
        gate[row * kColumns + column] = bfloat(v);
      });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
        v = float(bfloat(v)) * splash_silu(float(gate[row * kColumns + column]));
        output[out + ulong(row) * p.output_size + column] = bfloat(v);
      });
  };
  if constexpr (Rows == 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 16) run(integral_constant<uint, 2>{});
  else run(integral_constant<uint, 4>{});
}
using MoeExpertGgufMmaGateUpKernel =
    void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
         device uchar *, device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
         device uchar *, device uchar *, device uchar *, device uchar *, device uchar *, device bfloat *,
         device const uint *, constant MoeGgufGateUpParams &, uint2, uint, uint);
#define MOE_MMA_GATE_UP(R, S, suffix) \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up" suffix)]] \
      kernel MoeExpertGgufMmaGateUpKernel moe_expert_gguf_mma_gate_up<R, 1, S>; \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up_mixed" suffix)]] \
      kernel MoeExpertGgufMmaGateUpKernel moe_expert_gguf_mma_gate_up<R, 2, S>;
MOE_MMA_GATE_UP(8, 1, "") MOE_MMA_GATE_UP(32, 1, "")
MOE_MMA_GATE_UP(8, 2, "_s2") MOE_MMA_GATE_UP(32, 2, "_s2")
#undef MOE_MMA_GATE_UP
