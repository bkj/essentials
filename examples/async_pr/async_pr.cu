#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/async/pr.cuh>

using namespace gunrock;
using namespace memory;

void test_async_pr(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  // --
  // IO
  
  printf("io\n");
  
  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph
  
  printf("build graph\n");
  
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,
      csr.number_of_columns,
      csr.number_of_nonzeros,
      csr.row_offsets.data().get(),
      csr.column_indices.data().get(),
      csr.nonzero_values.data().get()
  );
  
  // --
  // Params and memory allocation
  
  weight_t lambda  = 0.85;
  weight_t epsilon = 0.01;
  vertex_t n_vertices    = G.get_number_of_vertices();
  thrust::device_vector<weight_t> rank(n_vertices);
  thrust::device_vector<weight_t> res(n_vertices);
  
  // --
  // Run problem
  
  printf("run\n");
  
  float elapsed = async::pr::run(G, lambda, epsilon, rank.data().get(), res.data().get());
  
  cudaDeviceSynchronize();
  printf("complete\n");
  
  // --
  // Log + Validate

  // std::cout << "Rank (output) = ";
  // thrust::copy(rank.begin(), rank.end(),
  //              std::ostream_iterator<weight_t>(std::cout, " "));
  // std::cout << std::endl;
  
  std::cout << "PR Elapsed Time: " << elapsed << " (ms)" << std::endl;
    
  thrust::host_vector<weight_t> h_rank = rank;
  weight_t acc = 0;
  for(vertex_t i = 0 ; i < n_vertices; i++) acc += h_rank[i];
  printf("acc=%f\n", acc);
  
}

int main(int argc, char** argv) {
  test_async_pr(argc, argv);
  return EXIT_SUCCESS;
}
