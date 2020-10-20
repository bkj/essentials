#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/runner.hxx>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

using namespace gunrock;
using namespace memory;

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
  
  constexpr memory::memory_space_t d_space = memory::memory_space_t::device;
  
  // --
  // IO
  
  std::string filename = argument_array[1];
  
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto coo = mm.load(filename);
  
  format::csr_t<d_space, vertex_t, edge_t, weight_t> csr;
  csr = coo;

  // --
  // Configure
    
  using graph_t = graph::graph_t<
      d_space, vertex_t, edge_t, weight_t,
      graph::graph_csr_t<d_space, vertex_t, edge_t, weight_t>>;
  
  using param_t   = sssp::sssp_param_t<graph_t>;
  using result_t  = sssp::sssp_result_t<graph_t>;
  using problem_t = sssp::sssp_problem_t<graph_t>;
  using enactor_t = sssp::sssp_enactor_t<problem_t>;
  
  vertex_t single_source = 0;
  param_t  param(single_source);
  result_t result(csr.number_of_rows);

  // --
  // Run
  
  float elapsed = csr_run<problem_t, enactor_t>(csr, param, result);

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
