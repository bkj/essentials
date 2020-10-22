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
  using weight_t = int;
  
  // --
  // IO
  
  std::string filename = argument_array[1];
  
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto coo = mm.load(filename);
  
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr = coo; 
  // ^^ Honestly don't love the operator overloading here -- it feels unexpected.  
  // I'd prefer `csr.load(coo)` or, even better, `csr = coo.tocsr()` or `csr = to_csr(coo)`

  // --
  // Build graph + metadata
  
  auto G    = graph::build::from_csr_t<memory_space_t::device>(&csr);
  auto meta = graph::build::meta_from_csr_t(&csr);
  
  using graph_t = decltype(G)::value_type;
  using meta_t  = decltype(meta)::value_type;
  
  // --
  // Setup problem
  
  using param_t   = sssp::sssp_param_t<meta_t>;
  using result_t  = sssp::sssp_result_t<meta_t>;
  using problem_t = sssp::sssp_problem_t<graph_t, meta_t>;
  using enactor_t = sssp::sssp_enactor_t<problem_t>;

  vertex_t single_source = 0;
  param_t  param(single_source);
  result_t result(meta.data()); // Don't love having to call `.data()`, but couldn't figure out how to work around

  float elapsed = run<problem_t, enactor_t>(G, meta, param, result);

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
