%module gunrock_sssp

%{
#define SWIG_FILE_WITH_INIT
#include "gunrock_sssp.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* IN_ARRAY1, int DIM1)   {(int* row_offsets, int n_row_offsets)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* column_indices, int n_column_indices)};
%apply (int* IN_ARRAY1, int DIM1)   {(int* nonzero_values, int n_nonzero_values)};
%apply (float* IN_ARRAY1, int DIM1) {(float* nonzero_values, int n_nonzero_values)};
%apply (float** ARGOUTVIEW_ARRAY1, int *DIM1) {(float** distances, int* n_distances)}

%include "gunrock_sssp.cuh"

%inline %{
template <typename vertex_t, typename edge_t, typename weight_t>
void _run_sssp(
  // --
  // Input
  vertex_t single_source, 
  edge_t* row_offsets, int n_row_offsets, 
  vertex_t* column_indices, int n_column_indices, 
  weight_t* nonzero_values, int n_nonzero_values,
  
  // --
  // Output
  
  weight_t** distances, int* n_distances
  
) {
  int n_nodes = n_row_offsets - 1;
  int n_edges = n_column_indices;
  run_sssp(single_source, distances, n_nodes, n_edges, row_offsets, column_indices, nonzero_values);
  *n_distances = n_nodes;
}
%}

%template(run_sssp_IIF) _run_sssp<int, int, float>;