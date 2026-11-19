#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/MoE.hpp"

namespace splash::test {

// One step of a MoE block as the runtime composes it (QwenTarget's layers):
// its routing, the load of a streamed block's step that runs around it
// (ops::MoE::aroundLoad) beside its resident experts' passes, then its passes
// over the rest, or all its routed work where it loads its experts itself,
// and its combine.
inline void addMoeStep(metal::CommandGraph &graph, const ops::MoeBuffers &buffers,
                       const ops::BlockMoeWeights &weights, const ops::MoePlan &plan) {
  ops::MoE::addRouting(graph, buffers, weights, plan);
  if (ops::MoE::aroundLoad(buffers, weights, plan)) {
    const ops::MoeLoadPart part{&buffers, &plan};
    weights.stream->addLoad(graph, {&part, 1}, weights.streamBlock,
                            [&] { ops::MoE::addResident(graph, buffers, weights, plan); });
  }
  ops::MoE::addRouted(graph, buffers, weights, plan);
}

} // namespace splash::test
