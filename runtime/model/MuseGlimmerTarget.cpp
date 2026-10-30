#include "model/MuseGlimmerTarget.hpp"

#include "model/MuseGlimmer.hpp"
#include "ops/Normalization.hpp"

#include <utility>

namespace splash::model {

MuseGlimmerTarget::MuseGlimmerTarget(const MuseGlimmerWeights &weights, const QwenTargetGeometry &geometry,
                                     metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  requireLayers(weights.layers.size(), weights.layers.size());
}

std::vector<metal::MetalBuffer> MuseGlimmerTarget::addPrefill(metal::CommandGraph &graph,
                                                              std::span<const QwenTargetPrefillChunk> chunks,
                                                              const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                              QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  return addPrefillLayers(std::span(steps), *this, layers);
}

void MuseGlimmerTarget::addPrefillBegin(PrefillStep &step) const {
  const float eps = static_cast<float>(geometry_.rmsEpsilon);
  ops::Normalization::addUnitRms(step.graph, step.buffers.hidden[0], step.buffers.hidden[0],
                                 geometry_.hiddenSize, step.rows, eps);
}

void MuseGlimmerTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const MuseGlimmerLayerWeights &layer = weights_.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1], output = b.hidden[(index & 1) ^ 1];
  const float eps = static_cast<float>(geometry_.rmsEpsilon);
  ops::Normalization::addRms(step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows, {},
                             ops::LinearInput::Plain, eps);
  addPrefillMixer(step, layer.mixer, {b.attentionOutput, {}});
  ops::Normalization::addRmsResidual(step.graph, b.attentionOutput, layer.postAttentionNorm, input,
                                     b.attentionOutput, geometry_.hiddenSize, step.rows, geometry_.postNormEpsilon);
  ops::Normalization::addRms(step.graph, b.attentionOutput, layer.ffnNorm, b.normalized, geometry_.hiddenSize,
                             step.rows, {}, ops::LinearInput::Plain, eps);
  operators_.linear().addPrefillSwiGlu(step.graph, layer.projections(), b.ffn(), {}, output, step.rows);
  ops::Normalization::addRmsResidual(step.graph, output, layer.postFfnNorm, b.attentionOutput, output,
                                     geometry_.hiddenSize, step.rows, geometry_.postNormEpsilon);
}

const ops::BlockMoeWeights *MuseGlimmerTarget::moe(uint32_t) const { return nullptr; }

void MuseGlimmerTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addCapture(step.graph, slot, b.hidden[(index & 1) ^ 1], source, b.captured, destination, rows);
  });
}

metal::MetalBuffer MuseGlimmerTarget::addPrefillEnd(PrefillStep &step) const {
  return step.buffers.hidden[geometry_.layers & 1];
}

void MuseGlimmerTarget::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                                  const kv::PageStorage &kv, SpilledKvReads &spilled,
                                  std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  VerifyStep step = verifyStep(graph, buffers, kv, spilled, chunks, lanes);
  const float eps = static_cast<float>(geometry_.rmsEpsilon);
  ops::Normalization::addUnitRms(step.graph, buffers.hidden[0], buffers.hidden[0], geometry_.hiddenSize, step.rows,
                                 eps);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const MuseGlimmerLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1], output = buffers.hidden[(index & 1) ^ 1];
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, layer.inputNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
        buffers.linearScratch, decodeInput(layer.mixer.inputProjection, step.lanes), eps);
    addVerifyMixer(step, layer.mixer, normalized, {buffers.attentionOutput, {}});
    ops::Normalization::addRmsResidual(step.graph, buffers.attentionOutput, layer.postAttentionNorm, input,
                                       buffers.attentionOutput, geometry_.hiddenSize, step.rows,
                                       geometry_.postNormEpsilon);
    const ops::Linear &linear = operators_.linear();
    const ops::LinearPlan gateUpPlan =
        linear.decodePlan(layer.upProjection, step.lanes, ops::LinearEpilogue::GateUp, &layer.gateProjection);
    const ops::PreparedInput ffnNormalized = ops::Normalization::addRms(
        step.graph, buffers.attentionOutput, layer.ffnNorm, buffers.normalized, geometry_.hiddenSize, step.rows,
        buffers.linearScratch, gateUpPlan.input(), eps);
    linear.add(step.graph,
               {.input = buffers.normalized, .output = buffers.denseIntermediate,
                .gateScratch = buffers.denseGateScratch, .scratch = buffers.linearScratch,
                .prepared = ffnNormalized},
               layer.upProjection, gateUpPlan, &layer.gateProjection);
    linear.add(step.graph,
               {.input = buffers.denseIntermediate, .output = output, .scratch = buffers.linearScratch},
               layer.downProjection, linear.decodePlan(layer.downProjection, step.lanes, ops::LinearEpilogue::None));
    ops::Normalization::addRmsResidual(step.graph, output, layer.postFfnNorm, buffers.attentionOutput, output,
                                       geometry_.hiddenSize, step.rows, geometry_.postNormEpsilon);
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, output, 0, buffers.capturedTargetHidden, 0, step.rows);
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, step.lanes,
               buffers.linearScratch);
}

void MuseGlimmerTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                     metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                                     ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), lanes,
                    scratch);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const MuseGlimmerWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<MuseGlimmerTarget>(weights, geometry, backend, operators);
}

} // namespace splash::model
