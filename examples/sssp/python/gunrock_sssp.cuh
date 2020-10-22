#ifndef PYSSSP_CUH
#define PYSSSP_CUH

#include <gunrock/applications/runner.hxx>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

using namespace gunrock;
using namespace memory;

// Nothing about this is specific to python -- a version of this could be used as an interface for all external APIs

template <typename vertex_t, typename edge_t, typename weight_t>
void run_sssp(vertex_t single_source, weight_t** distances, int n_nodes, int n_edges, edge_t* row_offsets, vertex_t* column_indices, weight_t* nonzero_values) {
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr(n_nodes, n_nodes, n_edges);
  
  // !! I'm assuming this does not copy data?
  thrust::host_vector<edge_t> h_row_offsets(row_offsets, row_offsets + n_nodes + 1);
  thrust::host_vector<vertex_t> h_column_indices(column_indices, column_indices + n_edges);
  thrust::host_vector<weight_t> h_nonzero_values(nonzero_values, nonzero_values + n_edges);
  
  csr.row_offsets    = h_row_offsets;
  csr.column_indices = h_column_indices;
  csr.nonzero_values = h_nonzero_values;
  
  // --
  // Build graph + metadata

  auto G    = graph::build::from_csr_t<memory_space_t::device>(&csr);
  auto meta = graph::build::meta_from_csr_t(&csr);
  
  using graph_t = typename decltype(G)::value_type;
  using meta_t  = typename decltype(meta)::value_type;

  // --
  // Setup problem

  using param_t   = sssp::sssp_param_t<meta_t>;
  using result_t  = sssp::sssp_result_t<meta_t>;
  using problem_t = sssp::sssp_problem_t<graph_t, meta_t>;
  using enactor_t = sssp::sssp_enactor_t<problem_t>;

  param_t  param(single_source);
  result_t result(meta.data());

  float elapsed = run<problem_t, enactor_t>(G, meta, param, result);

  // --
  // Log

  std::cout << "Distances (output) = ";
  thrust::copy(result.distances.begin(), result.distances.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
  
  // --
  // Return results
  
  *distances = (weight_t*)malloc(n_nodes * sizeof(weight_t));
  cudaMemcpy(*distances, thrust::raw_pointer_cast(result.distances.data()), n_nodes * sizeof(weight_t), cudaMemcpyDeviceToHost);
}

#endif
