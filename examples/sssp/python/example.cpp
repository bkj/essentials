#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #include "sssp.cuh"

namespace py = pybind11;

template <typename T>
struct Pet {
    T x;
    Pet(T x_) {
      x = x_;
    }
    
    T square() {
      return x * x;
    }
};

template <typename T>
Pet<T> __make_pet(T x) {
  return Pet<T>(x);
}

PYBIND11_MODULE(example, m) {
  //   using PetI = Pet<int>;
  //   py::class_<PetI>(m, "PetI")
  //       .def(py::init<int>())
  //       .def("square", &PetI::square)
  //       .def_readwrite("x", &PetI::x);

  //   using PetF = Pet<float>;
  //   py::class_<PetF>(m, "PetF")
  //       .def(py::init<int>())
  //       .def("square", &PetF::square)
  //       .def_readwrite("x", &PetF::x);
    
  // // m.def("make_pet", make_pet<int>);

  m.def("Pet", py::overload_cast<int>(&__make_pet<int>))
    .def("Pet", py::overload_cast<float>(&__make_pet<float>));
  
  // m.def("test_sssp", test_sssp);
}