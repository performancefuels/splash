// A layer-ending split must not submit GPU work ahead of an earlier host
// step. Reproduces streamed GGUF warm-up NaNs: the GPU reads staging before
// the CPU fills it when the split bypasses HostSteps::commitAfter.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestMethodReplacement.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/RowCopy.hpp"

#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>

namespace {
std::atomic<unsigned> committed{0};
std::promise<void> splitCompleted;
IMP originalCommit = nullptr;
void observeCommit(id command, SEL selector) {
  if (++committed == 2)
    [command addCompletedHandler:^(id<MTLCommandBuffer>) { splitCompleted.set_value(); }];
  reinterpret_cast<void (*)(id, SEL)>(originalCommit)(command, selector);
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: host_step_split_test <production.metallib>\n";
      return 2;
    }
    using namespace splash;
    metal::MetalBackend backend(argv[1]);
    auto source = test::sharedBuffer(backend, sizeof(uint16_t));
    auto middle = test::sharedBuffer(backend, sizeof(uint16_t));
    auto output = test::sharedBuffer(backend, sizeof(uint16_t));
    auto *value = static_cast<uint16_t *>(source.contents());
    *value = 0x3f00;
    metal::CommandGraph warm;
    ops::RowCopy::add(warm, source, {0, 1, 0}, middle, {0, 1, 0}, 1, 1);
    (void)backend.submitCommandAsync(warm.command()).wait();

    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto gate = release.get_future().share();
    auto earlySplit = splitCompleted.get_future();
    metal::CommandGraph graph;
    graph.host([&] {
      entered.set_value();
      gate.wait();
      *value = 0x3f80;
    });
    ops::RowCopy::add(graph, source, {0, 1, 0}, middle, {0, 1, 0}, 1, 1);
    graph.split();
    ops::RowCopy::add(graph, middle, {0, 1, 0}, output, {0, 1, 0}, 1, 1);
    id<MTLCommandBuffer> command = [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    test::MethodReplacement observe(command, @selector(commit), reinterpret_cast<IMP>(observeCommit));
    originalCommit = observe.original;
    auto ticket = backend.submitCommandAsync(graph.command());
    const bool hostEntered = started.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    // A broken split can finish while the host is blocked. Otherwise this
    // bounded wait releases the host so that the correct GPU copy can run.
    (void)earlySplit.wait_for(std::chrono::seconds(1));
    release.set_value();
    (void)ticket.wait();
    test::require(hostEntered, "host step did not start");
    test::require(*static_cast<uint16_t *>(middle.contents()) == 0x3f80 &&
                      *static_cast<uint16_t *>(output.contents()) == 0x3f80,
                  "GPU work ending at a split read data before its host step wrote it");

    // A four-lane streamed verify can need more than 512 buffers after the
    // beside-host overlap added in 1.6.1. Every host/GPU handoff must still
    // run in order without filling the queue before submission can finish.
    constexpr unsigned kSteps = 600;
    unsigned completed = 0;
    metal::CommandGraph chain;
    for (unsigned index = 0; index < kSteps; ++index) {
      chain.host([&, index] {
        if (index)
          test::require(*static_cast<uint16_t *>(output.contents()) == (index % 2 ? 0x3f00 : 0x3f80),
                        "a long host-step chain overtook its preceding GPU copy");
        *value = index % 2 ? 0x3f80 : 0x3f00;
        ++completed;
      });
      ops::RowCopy::add(chain, source, {0, 1, 0}, output, {0, 1, 0}, 1, 1);
    }
    (void)backend.submitCommandAsync(chain.command()).wait();
    test::require(completed == kSteps && *static_cast<uint16_t *>(output.contents()) == 0x3f80,
                  "a long streamed host-step chain did not finish");
    std::cout << "host_step_split_test: PASS\n";
  }
}
