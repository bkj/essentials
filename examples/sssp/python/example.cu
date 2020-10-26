#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "sssp.cuh"

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
  m.def("make_graph", graph::build::make_graph<memory_space_t::host, int, int, float>);
}