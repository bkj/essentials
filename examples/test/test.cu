#include <cstdlib>  // EXIT_SUCCESS
#include "nvToolsExt.h"
#include <gunrock/applications/application.hxx>

__global__ void kernel(int n, int* x) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i   = x[idx];
    int acc = 0;
    for(int ii = 0; ii < i; ii++) {
      acc += ii;
    }
    x[i] = (int)(acc % 2);
  }
}

void do_test(int num_arguments, char** argument_array) {
  
  // --
  // Create data
  
  int n = 200000;
  
  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);
  
  for(int i = 0; i < n; i++) h_input[i] = i;
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);
    
  // --
  // Setup
  
  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);
  
  cudaSetDevice(0);
  thrust::device_vector<int> input0  = h_input;
  thrust::device_vector<int> output0 = h_output;

  cudaSetDevice(1);
  thrust::device_vector<int> input1  = h_input;
  thrust::device_vector<int> output1 = h_output;

  cudaSetDevice(2);
  thrust::device_vector<int> input2  = h_input;
  thrust::device_vector<int> output2 = h_output;

  cudaSetDevice(3);
  thrust::device_vector<int> input3  = h_input;
  thrust::device_vector<int> output3 = h_output;

  std::vector<thrust::device_vector<int>*> all_inputs;
  all_inputs.push_back(&input0);
  all_inputs.push_back(&input1);
  all_inputs.push_back(&input2);
  all_inputs.push_back(&input3);
  
  std::vector<thrust::device_vector<int>*> all_outputs;
  all_outputs.push_back(&output0);
  all_outputs.push_back(&output1);
  all_outputs.push_back(&output2);
  all_outputs.push_back(&output3);

  cudaSetDevice(0);

  auto chunk_size = n / num_gpus;
  std::cout << "num_gpus  : " << num_gpus << std::endl;
  std::cout << "chunk_size: " << chunk_size << std::endl;
  
  struct gpu_info {
    cudaStream_t stream;
    cudaEvent_t  event;
  };
  
  std::vector<gpu_info> infos;
  
  cudaStream_t master_stream;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  
  for(int i = 0 ; i < num_gpus ; i++) {
    gpu_info info;
    
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreate(&info.event);
    
    infos.push_back(info);
  }
  
  // --
  // Run
  
  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for(int ii = 0; ii < i; ii++) {
      acc += ii;
    }
    return acc % 2 == 0;
  };
  
  nvtxRangePushA("work");
  
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);
    auto n_vals = thrust::copy_if(
      thrust::cuda::par.on(infos[i].stream),
      all_inputs[i]->begin(),
      all_inputs[i]->end(),
      all_outputs[i]->begin(),
      fn
    );
    // kernel<<<(n + 255) / 256, 256, 0, infos[i].stream>>>(
    //   n, 
    //   all_inputs[i]->data().get(),
    //   all_outputs[i]->data().get(),
    // );
    
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  printf("done run\n");
  
  for(int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  
  nvtxRangePop();
  
  // cudaSetDevice(0);
  // thrust::host_vector<int> tmp = *all_outputs[0];
  // thrust::copy(tmp.begin(), tmp.end(), std::ostream_iterator<int>(std::cout, " "));
  // std::cout << std::endl;
}

int main(int argc, char** argv) {
  do_test(argc, argv);
  return EXIT_SUCCESS;
}
