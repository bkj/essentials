#ifndef PYSSSP_CUH
#define PYSSSP_CUH

#include <gunrock/applications/runner.hxx>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

using namespace gunrock;
using namespace memory;

// Nothing about this is specific to python -- a version of this could be used as an interface for all external APIs

template <typename vertex_t, typename edge_t, typename weight_t>
void do_sssp(vertex_t single_source, edge_t* indptr, int n_indptr, vertex_t* indices, int n_indices, weight_t* data, int n_data) {
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr(n_indptr - 1, n_indptr - 1, n_indices);
  
  // I'm sure there's a better way to get data from host pointers to thrust::device_vector?
  // @neoblizz -- what's the right way to do this?
  thrust::host_vector<edge_t> h_indptr(n_indptr);
  thrust::host_vector<vertex_t> h_indices(n_indices);
  thrust::host_vector<weight_t> h_data(n_data);

  thrust::device_vector<edge_t> d_indptr(n_indptr);
  thrust::device_vector<vertex_t> d_indices(n_indices);
  thrust::device_vector<weight_t> d_data(n_data);
  
  for(int i = 0; i < n_indptr; i++)  h_indptr[i] = indptr[i];
  for(int i = 0; i < n_indices; i++) h_indices[i] = indices[i];
  for(int i = 0; i < n_data; i++)    h_data[i] = data[i];
  
  csr.row_offsets    = h_indptr;
  csr.column_indices = h_indices;
  csr.nonzero_values = h_data;
  
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
  thrust::copy(result.distances.begin(), result.distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " ")); // !! Helper function for printing vectors?
  std::cout << std::endl;
  std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

#endif
