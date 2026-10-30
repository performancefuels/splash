#include "model/MuseGlimmer.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {

MuseGlimmerWeights loadMuseGlimmerWeights(metal::MetalBackend &backend, MuseGlimmerLayout layout,
                                          const QwenTargetFiles &files) {
  return loadQwenTarget<MuseGlimmerWeights>(
      backend, layout, files, [&](GgufTargetLoader &gguf, const BlockTargetFormat &format, const MoeExperts &) {
        MuseGlimmerWeights result;
        result.layout = layout;
        result.layers.reserve(layout.layers);
        for (uint32_t index = 0; index < layout.layers; ++index) {
          WeightFile file = gguf.layer(index);
          MuseGlimmerLayerWeights &layer = result.layers.emplace_back();
          layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
          layer.mixer = readQwenAttention(file, format, layout);
          layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
          layer.ffnNorm = format.norm(file, layout.hiddenSize, "ffn-norm");
          layer.gateProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
          layer.upProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
          layer.downProjection = format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
          layer.postFfnNorm = format.norm(file, layout.hiddenSize, "post-ffn-norm");
          file.finish();
          result.files.push_back(file.record());
        }
        readQwenTargetHead(result, layout, gguf, format);
        return result;
      });
}

const ops::BlockMoeWeights *qwenMoeBlock(const MuseGlimmerLayerWeights &) noexcept { return nullptr; }

void includeFamilyProjections(QwenTargetGeometry &geometry, const MuseGlimmerWeights &weights) {
  for (const MuseGlimmerLayerWeights &layer : weights.layers)
    includeDenseFfn(geometry, layer.projections());
}

} // namespace splash::model
