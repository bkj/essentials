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

%apply (int* IN_ARRAY1, int DIM1)   {(int* indices, int n_indices)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* indptr, int n_indptr)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* data, int n_data)};
%apply (float* IN_ARRAY1, int DIM1) {(float* data, int n_data)};

%apply (int* IN_ARRAY1, int DIM1)   {(int* x, int n)};
%apply (float* IN_ARRAY1, int DIM1) {(float* y, int m)};

// %apply (int* IN_ARRAY1, int DIM1) {(int* x, int n), (int* y, int m)};

%include "test.cuh"

%template(do_testI) do_test<int>;
%template(do_testF) do_test<float>;

%template(do_test_ssspI) test_sssp<int, float>;
// %template(do_test_ssspF) test_sssp<float>;

// %template(do_sssp_III) do_sssp<int, int, int>;
%template(do_sssp_IIF) do_sssp<int, int, float>;