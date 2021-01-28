#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/async/apsp.cuh>

using namespace gunrock;
using namespace memory;

void test_async_apsp(int num_arguments, char** argument_array) {
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
  
  
  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph
  
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
  
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices * n_vertices);
  
  // --
  // Run problem
  
  
  float elapsed = async::apsp::run(G, distances.data().get());
  
  cudaDeviceSynchronize();
  
  // --
  // Log + Validate
  
  thrust::host_vector<edge_t> h_distances = distances;
  
  for(vertex_t i = 0 ; i < n_vertices; i++) {
    for(vertex_t j = 0; j < n_vertices; j++) {
      std::cout << h_distances[n_vertices * i + j] << " ";
    }
    std::cout << std::endl;
  }
  
  printf("\n");
  printf("elapsed=%f\n", elapsed);
}

int main(int argc, char** argv) {
  test_async_apsp(argc, argv);
  return EXIT_SUCCESS;
}
