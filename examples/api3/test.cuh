#ifndef PYCUREVERSE_CUH
#define PYCUREVERSE_CUH

#include <gunrock/applications/runner.hxx>

template <class T> 
T do_test() {
    return (T)(-1);
}

int do_another_test() {
    return -2;
}

float yet_another_test() {
    return -3.0;
}

template <typename T>
T test_sssp(T* x, int n, T* y, int m) {

  T z = 0;
  for(int i = 0; i < n; i++) {
      printf("%d ", x[i]);
      z += x[i];
  }
  printf("\n");

  for(int i = 0; i < m; i++) {
      printf("%d ", y[i]);
      z += y[i];
  }
  printf("\n");
  
  return z;
  
  // --
  // Define types

//   using vertex_t = int;
//   using edge_t   = int;
//   using weight_t = float;

//   thrust::device_vector<weight_t> test;
//   thrust::device_vector<weight_t> result(1);
//   test.resize(10);
//   thrust::fill(thrust::device, test.begin(), test.end(), 1);
  
//   result[0] = thrust::reduce(thrust::device, test.begin(), test.end(), 0);
//   return result[0];

  // // --
  // // IO

  // std::string filename = argument_array[1];

  // io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  // auto coo = mm.load(filename);

  // format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  // csr = coo;
  // // ^^ Honestly don't love the operator overloading here -- it feels unexpected.
  // // I'd prefer `csr.load(coo)` or, even better, `csr = coo.tocsr()` or `csr = to_csr(coo)`

  // // --
  // // Build graph + metadata

  // auto G    = graph::build::from_csr_t<memory_space_t::device>(&csr);
  // auto meta = graph::build::meta_from_csr_t(&csr);

  // using graph_t = decltype(G)::value_type;
  // using meta_t  = decltype(meta)::value_type;

  // // --
  // // Setup problem

  // using param_t   = sssp::sssp_param_t<meta_t>;
  // using result_t  = sssp::sssp_result_t<meta_t>;
  // using problem_t = sssp::sssp_problem_t<graph_t, meta_t>;
  // using enactor_t = sssp::sssp_enactor_t<problem_t>;

  // vertex_t single_source = 0;
  // param_t  param(single_source);
  // result_t result(meta.data());

  // float elapsed = run<problem_t, enactor_t>(G, meta, param, result);

  // // --
  // // Log

  // std::cout << "Distances (output) = ";
  // thrust::copy(result.distances.begin(), result.distances.end(),
  //              std::ostream_iterator<weight_t>(std::cout, " ")); // !! Helper function for printing vectors?
  // std::cout << std::endl;
  // std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

#endif
