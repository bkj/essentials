#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/application.hxx>

using namespace gunrock;
using namespace memory;

void test_read_binary(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/mtx2binary <inpath> <outpath>" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string inpath = argument_array[1];
  std::string outpath = argument_array[2];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t = format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr0;
  csr0.from_coo(mm.load(inpath));

  std::cout << "csr0.number_of_rows     = " << csr0.number_of_rows     << std::endl;
  std::cout << "csr0.number_of_columns  = " << csr0.number_of_columns  << std::endl;
  std::cout << "csr0.number_of_nonzeros = " << csr0.number_of_nonzeros << std::endl;
  
  csr0.write_binary(outpath);
  
  csr_t csr1;
  csr1.read_binary(outpath);
  std::cout << "csr1.number_of_rows     = " << csr1.number_of_rows     << std::endl;
  std::cout << "csr1.number_of_columns  = " << csr1.number_of_columns  << std::endl;
  std::cout << "csr1.number_of_nonzeros = " << csr1.number_of_nonzeros << std::endl;
  
  // --
  // Log
  
  std::cout << "----------------" << std::endl;
  
  thrust::copy(csr0.row_offsets.begin(), csr0.row_offsets.end(), std::ostream_iterator<edge_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "-" << std::endl;
  thrust::copy(csr1.row_offsets.begin(), csr1.row_offsets.end(), std::ostream_iterator<edge_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "----------------" << std::endl;
  
  thrust::copy(csr0.column_indices.begin(), csr0.column_indices.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "-" << std::endl;
  thrust::copy(csr1.column_indices.begin(), csr1.column_indices.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
  std::cout << std::endl;
  
  std::cout << "----------------" << std::endl;
}

int main(int argc, char** argv) {
  test_read_binary(argc, argv);
  return EXIT_SUCCESS;
}
