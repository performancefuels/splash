#pragma once

// Muse Glimmer 30B's target (model/MuseGlimmer.hpp): 52 dense attention layers
// with parameterless embedding RMSNorm before layer 0, pre/post attention RMS
// norms, pre/post FFN RMS norms, and scaled/softcapped logits.

#include "model/QwenTarget.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

struct MuseGlimmerLayout;
struct MuseGlimmerLayerWeights;

class MuseGlimmerTarget final : public QwenTarget {
public:
  MuseGlimmerTarget(const QwenTargetWeights<MuseGlimmerLayout, MuseGlimmerLayerWeights> &weights,
                    const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                    const ops::ExecutionPlans &operators);

  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const override;
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override;

private:
  friend class QwenTarget;
  void addPrefillBegin(PrefillStep &step) const;
  [[nodiscard]] bool attends(uint32_t) const { return true; }
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  const QwenTargetWeights<MuseGlimmerLayout, MuseGlimmerLayerWeights> &weights_;
};

} // namespace splash::model
