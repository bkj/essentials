%module test

%{
/* Includes the header in the wrapper code */
#define SWIG_FILE_WITH_INIT
#include "test.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* IN_ARRAY1, int DIM1) {(int* x, int n), (int* y, int m)};
%apply (float* IN_ARRAY1, int DIM1) {(float* x, int n), (float* y, int m)};

%include "test.cuh"

%template(do_testI) do_test<int>;
%template(do_testF) do_test<float>;

%template(do_test_ssspI) test_sssp<int>;
%template(do_test_ssspF) test_sssp<float>;
