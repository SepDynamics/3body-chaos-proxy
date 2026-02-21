#include "symbolic_chaos.h"

#include <atomic>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chaos_proxy {

bool ChaosEvent::operator==(const ChaosEvent &other) const {
  return index == other.index && state == other.state &&
         bit_prev == other.bit_prev && bit_curr == other.bit_curr;
}

std::vector<ChaosEvent> transform_rich(const std::vector<uint8_t> &bits) {
  if (bits.size() < 2)
    return {};

  const std::size_t n = bits.size() - 1;
  std::vector<ChaosEvent> result(n);

  // Index space [0, n)
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  std::atomic<bool> invalid{false};

  // Calculate transition dynamics in parallel
  std::for_each(
      std::execution::par_unseq, idx.begin(), idx.end(), [&](std::size_t i) {
        uint8_t prev = bits[i];
        uint8_t curr = bits[i + 1];
        if ((prev != 0 && prev != 1) || (curr != 0 && curr != 1)) {
          invalid.store(true, std::memory_order_relaxed);
          return;
        }

        ChaosState st = ChaosState::LOW_FLUCTUATION;
        if ((prev == 0 && curr == 1) || (prev == 1 && curr == 0)) {
          st = ChaosState::OSCILLATION; // Oscillation -> normal dynamic
                                        // adjustment
        } else if (prev == 1 && curr == 1) {
          st = ChaosState::PERSISTENT_HIGH; // Persistent high variance ->
                                            // instability
        }
        result[i] = ChaosEvent{static_cast<uint32_t>(i), st, prev, curr};
      });

  if (invalid.load(std::memory_order_relaxed))
    return {};
  return result;
}

std::vector<ChaosAggregateEvent>
aggregate(const std::vector<ChaosEvent> &events) {
  if (events.empty())
    return {};
  std::vector<ChaosAggregateEvent> aggregated;
  aggregated.push_back({events[0].index, events[0].state, 1});
  for (size_t i = 1; i < events.size(); ++i) {
    if (events[i].state == aggregated.back().state) {
      aggregated.back().count++;
    } else {
      aggregated.push_back({events[i].index, events[i].state, 1});
    }
  }
  return aggregated;
}

std::optional<ChaosState> ChaosProcessor::process(uint8_t current_bit) {
  if (current_bit != 0 && current_bit != 1)
    return std::nullopt;
  if (!prev_bit.has_value()) {
    prev_bit = current_bit;
    return std::nullopt;
  }
  uint8_t prev = prev_bit.value();
  std::optional<ChaosState> event_state;
  if (prev == 0 && current_bit == 0) {
    event_state = ChaosState::LOW_FLUCTUATION;
  } else if ((prev == 0 && current_bit == 1) ||
             (prev == 1 && current_bit == 0)) {
    event_state = ChaosState::OSCILLATION;
  } else if (prev == 1 && current_bit == 1) {
    event_state = ChaosState::PERSISTENT_HIGH;
  }
  prev_bit = current_bit;
  return event_state;
}

void ChaosProcessor::reset() { prev_bit.reset(); }

ChaosBasedProcessor::ChaosBasedProcessor(const ChaosOptions &options)
    : options_(options) {}

ChaosResult ChaosBasedProcessor::analyze(const std::vector<uint8_t> &bits) {
  ChaosResult result;
  result.collapse_threshold = options_.collapse_threshold;

  // Transform bits to events mapping to symbolic dynamics
  result.events = transform_rich(bits);
  result.aggregated_events = aggregate(result.events);

  // Count event types
  for (const auto &event : result.events) {
    if (event.state == ChaosState::LOW_FLUCTUATION) {
      result.low_fluctuation_count++;
    } else if (event.state == ChaosState::OSCILLATION) {
      result.oscillation_count++;
    } else if (event.state == ChaosState::PERSISTENT_HIGH) {
      result.persistent_high_count++;
    }
  }

  // Calculate fundamental macroscopic metrics
  if (!result.events.empty()) {
    // Fluctuation Persistence = fraction of time the structural fluctuation
    // remains consistently high
    result.fluctuation_persistence =
        static_cast<float>(result.persistent_high_count) /
        static_cast<float>(result.events.size());

    result.oscillation_ratio = static_cast<float>(result.oscillation_count) /
                               static_cast<float>(result.events.size());

    float null_ratio = static_cast<float>(result.low_fluctuation_count) /
                       static_cast<float>(result.events.size());

    // Calculate Information Entropy across the symbolic dynamics mappings
    auto safe_log2 = [](float x) -> float {
      return (x > 0.0f) ? std::log2(x) : 0.0f;
    };

    result.entropy =
        -(null_ratio * safe_log2(null_ratio) +
          result.oscillation_ratio * safe_log2(result.oscillation_ratio) +
          result.fluctuation_persistence *
              safe_log2(result.fluctuation_persistence));

    // Normalize entropy to [0,1]
    result.entropy = std::fmax(0.05f, std::fmin(1.0f, result.entropy / 1.585f));

    float pattern_coherence = 1.0f - result.entropy;
    float stability_factor = 1.0f - result.fluctuation_persistence;
    float consistency_factor = 1.0f - result.oscillation_ratio;

    // Baseline sequence stability (excluding generic/external enhancements)
    result.coherence = pattern_coherence * 0.6f + stability_factor * 0.3f +
                       consistency_factor * 0.1f;
    result.coherence = std::fmax(0.01f, std::fmin(0.99f, result.coherence));
  }

  // Detect collapse
  result.collapse_detected =
      (result.fluctuation_persistence >= options_.collapse_threshold);

  return result;
}

bool ChaosBasedProcessor::detectCollapse(const ChaosResult &result) const {
  return result.collapse_detected ||
         result.fluctuation_persistence >= options_.collapse_threshold;
}

void ChaosBasedProcessor::reset() {
  ChaosProcessor::reset();
  current_state_ = ChaosState::LOW_FLUCTUATION;
  prev_bit_ = 0;
}

} // namespace chaos_proxy
