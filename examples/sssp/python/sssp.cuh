#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/runner.hxx>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

using namespace gunrock;
using namespace memory;

int test_sssp(int x) {
  return x + 100;
  // // --
  // // Define types
  
  // using vertex_t = int;
  // using edge_t   = int;
  // using weight_t = float;
  
  // // --
  // // IO
  
  // std::string filename = argument_array[1];
  
  // io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  // auto coo = mm.load(filename);
  
  // format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  // csr = coo; 

  // // --
  // // Build graph + metadata
  
  // auto G    = graph::build::from_csr_t<memory_space_t::device>(&csr);
  // auto meta = graph::build::meta_from_csr_t(&csr);
  
  // using graph_t = decltype(G)::value_type;
  // using meta_t  = decltype(meta)::value_type;

  // sssp::sssp_runner_t<graph_t, meta_t> runner(G.data().get(), meta.data(), 0);
  // float elapsed = runner.run();

  // // --
  // // Log
  
  // std::cout << "Distances (output) = ";
  // thrust::copy(runner.result.distances.begin(), runner.result.distances.end(),
  //              std::ostream_iterator<weight_t>(std::cout, " ")); // !! Helper function for printing vectors?
  // std::cout << std::endl;
  // std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}
