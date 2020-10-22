%module gunrock_sssp

%{
#define SWIG_FILE_WITH_INIT
#include "gunrock_sssp.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* IN_ARRAY1, int DIM1)   {(int* indices, int n_indices)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* indptr, int n_indptr)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* data, int n_data)};
%apply (float* IN_ARRAY1, int DIM1) {(float* data, int n_data)};

%include "gunrock_sssp.cuh"

%template(do_sssp_IIF) do_sssp<int, int, float>;