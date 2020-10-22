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

%include "test.cuh"

%template(do_sssp_IIF) do_sssp<int, int, float>;