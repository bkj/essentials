#include <cstdlib>  // EXIT_SUCCESS
#include "nvToolsExt.h"
#include <gunrock/applications/application.hxx>

void do_test(int num_arguments, char** argument_array) {
  
  // --
  // Create data
  
  int n = 1000000;
  thrust::host_vector<int> h_input;
  for(int i = 0; i < n; i++) {
    h_input.push_back(i);
  }
  
  thrust::device_vector<int> input = h_input;
  thrust::device_vector<int> output(input.size());
  
  thrust::fill(thrust::device, output.begin(), output.end(), -1);
  
  // --
  // Run filter
  
  int num_gpus = -1;
  cudaGetDeviceCount(&num_gpus);
  auto chunk_size = n / num_gpus;
  std::cout << "num_gpus  : " << num_gpus << std::endl;
  std::cout << "chunk_size: " << chunk_size << std::endl;
  
  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for(int ii = 0; ii < i; ii++) {
      acc += ii;
    }
    return acc % 2 == 0;
  };
  
  nvtxRangePushA("work");
  
  // int new_size = 0;
  for(int i = 0 ; i < num_gpus ; i++) {
    auto n_vals = thrust::copy_if(
      thrust::device,
      input.begin()  + (chunk_size * i),
      input.begin()  + (chunk_size * (i + 1)),
      output.begin() + (chunk_size * i),
      fn
    );
    // new_size += (int)thrust::distance(output.begin(), n_vals);  
  }
  
  nvtxRangePop();
  
  // thrust::copy(output.begin(), output.end(), std::ostream_iterator<int>(std::cout, " "));
  // std::cout << std::endl;
}

int main(int argc, char** argv) {
  do_test(argc, argv);
  return EXIT_SUCCESS;
}
