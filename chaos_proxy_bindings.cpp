#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "symbolic_chaos.h"

namespace py = pybind11;

struct Metrics {
  float coherence;
  float stability;
  float entropy;
  float fluctuation_persistence;
  std::uint16_t sig_c;
  std::uint16_t sig_s;
  std::uint16_t sig_e;
};

static std::uint16_t bucket(float value) {
  value = std::fmax(0.0f, std::fmin(1.0f, value));
  return static_cast<std::uint16_t>(std::lround(value * 1000.0f));
}

static void bytes_to_bits(const std::string &bytes,
                          std::vector<std::uint8_t> &bits) {
  bits.clear();
  bits.reserve(bytes.size() * 8);
  for (unsigned char byte : bytes) {
    for (int shift = 7; shift >= 0; --shift) {
      bits.push_back(static_cast<std::uint8_t>((byte >> shift) & 0x1U));
    }
  }
}

Metrics analyze_bits_native(const std::vector<std::uint8_t> &bits) {
  using namespace chaos_proxy;
  static ChaosOptions options;
  static ChaosBasedProcessor processor(options);
  processor.reset();
  auto result = processor.analyze(bits);

  // Abstract the core physical properties out from the event engine
  const float coherence = static_cast<float>(result.coherence);
  const float entropy = static_cast<float>(result.entropy);
  const float fluctuation_persistence =
      static_cast<float>(result.fluctuation_persistence);
  const float stability = 1.0f - fluctuation_persistence;

  Metrics metrics{};
  metrics.coherence = coherence;
  metrics.stability = stability;
  metrics.entropy = entropy;
  metrics.fluctuation_persistence = fluctuation_persistence;

  metrics.sig_c = bucket(coherence);
  metrics.sig_s = bucket(stability);
  metrics.sig_e = bucket(entropy);
  return metrics;
}

chaos_proxy::ChaosResult
analyze_bits_detailed(const std::vector<std::uint8_t> &bits) {
  using namespace chaos_proxy;
  static ChaosOptions options;
  static ChaosBasedProcessor processor(options);
  processor.reset();
  return processor.analyze(bits);
}

std::vector<Metrics> analyze_window_batch(py::sequence windows) {
  const size_t count = static_cast<size_t>(py::len(windows));
  std::vector<std::string> buffers;
  buffers.reserve(count);
  for (const py::handle &obj : windows) {
    buffers.emplace_back(py::cast<std::string>(obj));
  }

  std::vector<Metrics> results;
  results.reserve(buffers.size());
  if (buffers.empty()) {
    return results;
  }

  std::vector<std::uint8_t> bits;
  for (const std::string &buffer : buffers) {
    bytes_to_bits(buffer, bits);
    results.push_back(analyze_bits_native(bits));
  }
  return results;
}

PYBIND11_MODULE(chaos_proxy, m) {
  py::enum_<chaos_proxy::ChaosState>(m, "ChaosState")
      .value("LOW_FLUCTUATION", chaos_proxy::ChaosState::LOW_FLUCTUATION)
      .value("OSCILLATION", chaos_proxy::ChaosState::OSCILLATION)
      .value("PERSISTENT_HIGH", chaos_proxy::ChaosState::PERSISTENT_HIGH)
      .export_values();

  py::class_<chaos_proxy::ChaosEvent>(m, "ChaosEvent")
      .def_readonly("index", &chaos_proxy::ChaosEvent::index)
      .def_readonly("state", &chaos_proxy::ChaosEvent::state)
      .def_readonly("bit_prev", &chaos_proxy::ChaosEvent::bit_prev)
      .def_readonly("bit_curr", &chaos_proxy::ChaosEvent::bit_curr);

  py::class_<chaos_proxy::ChaosAggregateEvent>(m, "ChaosAggregateEvent")
      .def_readonly("index", &chaos_proxy::ChaosAggregateEvent::index)
      .def_readonly("state", &chaos_proxy::ChaosAggregateEvent::state)
      .def_readonly("count", &chaos_proxy::ChaosAggregateEvent::count);

  py::class_<chaos_proxy::ChaosResult>(m, "ChaosResult")
      .def_readonly("coherence", &chaos_proxy::ChaosResult::coherence)
      .def_readonly("stability", &chaos_proxy::ChaosResult::stability)
      .def_readonly("confidence", &chaos_proxy::ChaosResult::confidence)
      .def_readonly("collapse_detected",
                    &chaos_proxy::ChaosResult::collapse_detected)
      .def_readonly("fluctuation_persistence",
                    &chaos_proxy::ChaosResult::fluctuation_persistence)
      .def_readonly("collapse_threshold",
                    &chaos_proxy::ChaosResult::collapse_threshold)
      .def_readonly("low_fluctuation_count",
                    &chaos_proxy::ChaosResult::low_fluctuation_count)
      .def_readonly("oscillation_count",
                    &chaos_proxy::ChaosResult::oscillation_count)
      .def_readonly("persistent_high_count",
                    &chaos_proxy::ChaosResult::persistent_high_count)
      .def_readonly("oscillation_ratio",
                    &chaos_proxy::ChaosResult::oscillation_ratio)
      .def_readonly("entropy", &chaos_proxy::ChaosResult::entropy)
      .def_readonly("events", &chaos_proxy::ChaosResult::events)
      .def_readonly("aggregated_events",
                    &chaos_proxy::ChaosResult::aggregated_events);

  py::class_<Metrics>(m, "Metrics")
      .def_readonly("coherence", &Metrics::coherence)
      .def_readonly("stability", &Metrics::stability)
      .def_readonly("entropy", &Metrics::entropy)
      .def_readonly("fluctuation_persistence",
                    &Metrics::fluctuation_persistence)
      .def_readonly("sig_c", &Metrics::sig_c)
      .def_readonly("sig_s", &Metrics::sig_s)
      .def_readonly("sig_e", &Metrics::sig_e);

  m.def("analyze_bits", &analyze_bits_native,
        "Analyze a window of bits using the native manifold kernel");
  m.def("analyze_window", &analyze_bits_detailed,
        "Return the full native Chaos result for a bit window");
  m.def("analyze_window_batch", &analyze_window_batch, py::arg("windows"),
        "Analyze multiple byte windows in a single native pass");
  m.def("transform_rich", &chaos_proxy::transform_rich,
        "Transform bits into Chaos events");
  m.def("aggregate_events", &chaos_proxy::aggregate,
        "Aggregate consecutive Chaos events");
}
