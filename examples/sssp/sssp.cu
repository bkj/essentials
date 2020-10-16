#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/runner.hxx>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

using namespace gunrock;

void test_sssp(int num_arguments, char** argument_array) {
  
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }
  
  // --
  // Define types
  
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;
  
  constexpr memory::memory_space_t h_space = memory::memory_space_t::host;
  constexpr memory::memory_space_t d_space = memory::memory_space_t::device;
  
  // --
  // IO
  
  std::string filename = argument_array[1];
  
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto coo = mm.load(filename);
  
  format::csr_t<h_space, vertex_t, edge_t, weight_t> h_csr;
  format::csr_t<d_space, vertex_t, edge_t, weight_t> d_csr;
  
  h_csr = coo;
  
  // !! Helper function for copying `h_csr` to `d_car`?
  d_csr.number_of_rows     = h_csr.number_of_rows;     // !! IMO change to `n_rows`, `num_rows`
  d_csr.number_of_columns  = h_csr.number_of_columns;
  d_csr.number_of_nonzeros = h_csr.number_of_nonzeros;
  d_csr.row_offsets        = h_csr.row_offsets;        // !! IMO change to `offsets`, `indices` and `values`
  d_csr.column_indices     = h_csr.column_indices;
  d_csr.nonzero_values     = h_csr.nonzero_values;

  // --
  // Run
  
  using h_graph_t = graph::graph_t<
      h_space, vertex_t, edge_t, weight_t,
      graph::graph_csr_t<h_space, vertex_t, edge_t, weight_t>>;
  
  using d_graph_t = graph::graph_t<
      d_space, vertex_t, edge_t, weight_t,
      graph::graph_csr_t<d_space, vertex_t, edge_t, weight_t>>;

  // ========================= Nothing above this line has to change =========================
  using param_t   = sssp::sssp_param_t<d_graph_t, h_graph_t>;
  using result_t  = sssp::sssp_result_t<d_graph_t, h_graph_t>;
  using problem_t = sssp::sssp_problem_t<d_graph_t, h_graph_t>;
  using enactor_t = sssp::sssp_enactor_t<problem_t>;
  
  vertex_t single_source = 0;
  param_t param(single_source);
  
  result_t result(h_csr.number_of_rows);

  float elapsed = csr_run<problem_t, enactor_t, param_t, result_t>(
    h_csr,
    d_csr,
    param,
    result
  );

  // --
  // Log
  
  std::cout << "Distances (output) = ";
  thrust::copy(result.distances.begin(), result.distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " ")); // !! Helper function for printing vectors?
  std::cout << std::endl;
  std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
  return EXIT_SUCCESS;
}
