#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/bfs.hxx>

using namespace gunrock;
using namespace memory;

void test_bfs(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));
  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // --
  // Build graph + metadata

  auto G = graph::build::from_csr<memory_space_t::device,
                                  graph::view_t::csr | graph::view_t::csc>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get(),  // values
      row_indices.data().get(),         // row_indices
      column_offsets.data().get()       // column_offsets
  );

  // --
  // Params and memory allocation

  vertex_t single_source = 0;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // --
  // Run problem

  float elapsed = gunrock::bfs::run(G, single_source, distances.data().get(),
                                    predecessors.data().get());

  // --
  // Log

  std::cout << "Distances (output) = ";
  thrust::copy(distances.begin(), distances.end(),
               std::ostream_iterator<vertex_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "BFS Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
  return EXIT_SUCCESS;
}
