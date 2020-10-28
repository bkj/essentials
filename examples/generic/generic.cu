#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp/sssp_implementation.hxx>
#include <gunrock/applications/color/color_implementation.hxx>

using namespace gunrock;
using namespace memory;

void do_test(int num_arguments, char** argument_array) {
  
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/generic <algorithm-name> filename.mtx" << std::endl;
    exit(1);
  }
  
  // --
  // Define types
  
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;
  
  // --
  // IO
  
  std::string app_name = argument_array[1];
  std::string filename = argument_array[2];
  
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename)); 

  // --
  // Build graph + metadata
  
  auto [G, meta] = graph::build::from_csr_t<memory_space_t::device>(&csr);
  
  // --
  // Params and memory allocation
  
  float elapsed;
  if(app_name == "sssp") {
    // Params + Output
    vertex_t single_source = 0;
    vertex_t n_vertices = meta[0].get_number_of_vertices();
    thrust::device_vector<weight_t> distances(n_vertices);
    thrust::device_vector<vertex_t> predecessors(n_vertices);
    
    // Run
    elapsed = gunrock::sssp::run(
      G,
      meta,
      single_source,
      distances.data().get(),
      predecessors.data().get()    
    );
    
    // Log
    std::cout << "Distances (output) = ";
    thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<weight_t>(std::cout, " "));
    
  } else if (app_name == "color") {
    // Params + Output
    vertex_t n_vertices = meta[0].get_number_of_vertices();
    thrust::device_vector<vertex_t> colors(n_vertices);
    
    // Run
    elapsed = gunrock::color::run(
      G,
      meta,
      colors.data().get()    
    );
    
    // Log
    std::cout << "Colors (output) = ";
    thrust::copy(colors.begin(), colors.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  } else {
    std::cout << "!! Unknown Algorithm ";
  }

  std::cout << std::endl;
  std::cout << "Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  do_test(argc, argv);
  return EXIT_SUCCESS;
}
