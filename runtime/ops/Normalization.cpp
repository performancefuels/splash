#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <utility>
#include <stdexcept>

namespace splash::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  requireBytes(weights.buffer, weights.bytes(width), "norm weight");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout, float eps) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  struct Params {
    uint32_t width;
    float eps;
  } const params{width, eps};
  if (layout != LinearInput::Plain) {
    requireTableScratch(scratch, layout, width, rows);
    graph.add(normKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, params, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(normKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              params, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              params, {rows, 1, 1});
  return {};
}

void Normalization::addUnitRms(metal::CommandGraph &graph, metal::MetalBuffer input,
                               metal::MetalBuffer output, uint32_t width, uint32_t rows,
                               float eps) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  struct Params {
    uint32_t width;
    float eps;
  } const params{width, eps};
  graph.add("norm_rms_unit", {std::move(input), std::move(output)}, params, {rows, 1, 1});
}

void Normalization::addRmsResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
                                   const NormWeights &weight, metal::MetalBuffer residual,
                                   metal::MetalBuffer output, uint32_t width, uint32_t rows,
                                   float eps) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(residual, bytes, "norm residual");
  requireBytes(output, bytes, "norm output");
  struct Params {
    uint32_t width;
    float eps;
  } const params{width, eps};
  graph.add(normKernel("norm_rms_residual", weight, width),
            {std::move(input), weight.buffer, std::move(residual), std::move(output)},
            params, {rows, 1, 1});
}

void Normalization::addLogitSoftcap(metal::CommandGraph &graph, metal::MetalBuffer logits,
                                    uint32_t count, float scale, float softcap) {
  if (scale == 0.0F && softcap <= 0.0F) return;
  requireBytes(logits, uint64_t{count} * sizeof(float), "logits");
  struct Params {
    uint32_t count;
    float scale;
    float softcap;
  } const params{count, scale != 0.0F ? scale : 1.0F, softcap};
  const uint64_t groups = (uint64_t{count} + metal::CommandGraph::kDefaultThreads - 1) /
                          metal::CommandGraph::kDefaultThreads;
  graph.add("logit_scale_softcap", {std::move(logits)}, params, {groups, 1, 1});
}

} // namespace splash::ops
