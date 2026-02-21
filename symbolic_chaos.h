#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace chaos_proxy {

/**
 * Chaos State enumeration
 * In the context of 3-body physics, we primarily care about sequences of
 * bitstreams: 00 -> LOW_FLUCTUATION (Quiet) 01 or 10 -> OSCILLATION
 * (Uncertain/Oscillating) 11 -> PERSISTENT_HIGH (Persistent High Variance)
 */
enum class ChaosState { LOW_FLUCTUATION, OSCILLATION, PERSISTENT_HIGH };

struct ChaosEvent {
  uint32_t index{0};
  ChaosState state{ChaosState::LOW_FLUCTUATION};
  uint8_t bit_prev{0};
  uint8_t bit_curr{0};
  bool operator==(const ChaosEvent &other) const;
};

std::vector<ChaosEvent> transform_rich(const std::vector<uint8_t> &bits);

struct ChaosAggregateEvent {
  uint32_t index{0};
  ChaosState state{ChaosState::LOW_FLUCTUATION};
  uint32_t count{1};
};

std::vector<ChaosAggregateEvent>
aggregate(const std::vector<ChaosEvent> &events);

struct ChaosOptions {
  double collapse_threshold = 0.5;
  double entropy_weight = 0.30;
  double coherence_weight = 0.20;
};

struct ChaosResult {
  double coherence = 0.0;
  double stability = 0.0;
  double confidence = 0.0;
  bool collapse_detected = false;
  double fluctuation_persistence = 0.0;
  std::vector<ChaosEvent> events;

  double collapse_threshold = 0.5;
  std::vector<ChaosAggregateEvent> aggregated_events;
  uint32_t low_fluctuation_count = 0;
  uint32_t oscillation_count = 0;
  uint32_t persistent_high_count = 0;
  double oscillation_ratio = 0.0;
  double entropy = 0.0;
};

class ChaosProcessor {
public:
  ChaosProcessor() = default;
  virtual ~ChaosProcessor() = default;
  virtual std::optional<ChaosState> process(uint8_t current_bit);
  virtual void reset();

protected:
  std::optional<uint8_t> prev_bit;
};

class ChaosBasedProcessor : public ChaosProcessor {
public:
  explicit ChaosBasedProcessor(const ChaosOptions &options);
  ~ChaosBasedProcessor() override = default;
  ChaosResult analyze(const std::vector<uint8_t> &data);
  void reset() override;

  bool detectCollapse(const ChaosResult &result) const;

private:
  ChaosOptions options_;
  ChaosState current_state_ = ChaosState::LOW_FLUCTUATION;
  uint32_t prev_bit_ = 0;
};

} // namespace chaos_proxy
