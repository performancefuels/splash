#pragma once

// Muse Glimmer 30B (meta-models/Muse-Glimmer-30B; GGUF architecture
// muse-glimmer, model type muse_glimmer_text): a 52-layer dense transformer
// whose layers all attend, 3 sliding-window layers (window 2048, RoPE theta
// 500,000 over whole 128-dim heads) followed by 1 full-attention layer (NoPE),
// repeating, with 32 query heads over 2 KV heads and a sigmoid attention
// output gate before o_proj. Each layer applies an RMS norm before and after
// attention and before and after its dense SwiGLU FFN (pre-norms eps 1e-5,
// post-norms eps 1e-8), and a parameterless RMS norm (eps 1e-5) normalizes the
// token/vision embeddings entering layer 0. The head scales logits by
// output_multiplier (1/sqrt(26)) and softcaps them with 20 * tanh(x / 20).

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace splash::model {

// Muse Glimmer's sliding layers, a bit per layer: every layer except every
// fourth (3, 7, 11, ..., 51), which attend every key without RoPE.
inline constexpr uint64_t kMuseGlimmerSlidingLayers = [] {
  uint64_t layers = 0;
  for (uint32_t layer = 0; layer < 52; ++layer)
    if ((layer + 1) % 4 != 0) layers |= uint64_t{1} << layer;
  return layers;
}();

struct MuseGlimmerLayout final : QwenHybridLayout {
  static constexpr std::string_view modelType = "muse_glimmer_text";
  static constexpr bool flatConfig = false;
  static constexpr std::string_view ggufArchitecture = "muse-glimmer";
  static constexpr bool loadsSafetensors = false;

  constexpr MuseGlimmerLayout()
      : QwenHybridLayout{{.nativeContextTokens = 131'072,
                          .layers = 52,
                          .hiddenSize = 6656,
                          .vocabularySize = 202048,
                          .packedFullWidth = 8704,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 32,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 128,
                          .attentionGate = true,
                          .rotaryPairs = 64,
                          .rotaryTheta = 500'000.0F,
                          .slidingWindow = 2048,
                          .slidingLayers = kMuseGlimmerSlidingLayers,
                          .unrotatedLayers = ~kMuseGlimmerSlidingLayers & ((uint64_t{1} << 52) - 1),
                          .fullAttentionPeriod = 1,
                          .maskToken = 201818,
                          .firstDraftMaskToken = 200002,
                          .lastDraftMaskToken = 202047,
                          .stopTokens = {200001, 200008},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 19968,
                          .rmsEpsilon = 1e-5,
                          .postNormEpsilon = 1e-8F,
                          .logitScale = 0.19611613513818404F,
                          .finalLogitSoftcap = 20.0F},
                         "Muse-Glimmer-30B"} {}
  constexpr explicit MuseGlimmerLayout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const MuseGlimmerLayout &) const = default;
};

struct MuseGlimmerLayerWeights final {
  ops::NormWeights inputNorm;
  QwenAttentionWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::NormWeights ffnNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
  ops::NormWeights postFfnNorm;

  [[nodiscard]] ops::SwiGluProjections projections() const noexcept {
    return {&gateProjection, &upProjection, &downProjection};
  }
};

using MuseGlimmerWeights = QwenTargetWeights<MuseGlimmerLayout, MuseGlimmerLayerWeights>;

[[nodiscard]] MuseGlimmerWeights loadMuseGlimmerWeights(metal::MetalBackend &backend, MuseGlimmerLayout layout,
                                                        const QwenTargetFiles &files);

[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const MuseGlimmerLayerWeights &layer) noexcept;
void includeFamilyProjections(QwenTargetGeometry &geometry, const MuseGlimmerWeights &weights);
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const MuseGlimmerWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);

} // namespace splash::model
