#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace predicated {

#define MGPU

#ifndef MGPU
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Allocate output size if necessary.
  if (output->size() != input->size()) {
    output->resize(input->size());
  }

  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? op(i) : false;
  };

  // Copy w/ predicate!
  auto new_length = thrust::copy_if(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: begin
      input->end(),                            // input iterator: end
      output->begin(),                         // output iterator
      predicate                                // predicate
  );

  auto new_size = thrust::distance(output->begin(), new_length);
  
  size_t new_sizes[1];
  
  new_sizes[0] = new_size;
  
  output->resize(new_sizes[0]);

  // Uniquify!
  // auto new_end = thrust::unique(
  //     thrust::cuda::par.on(context.stream()),  // execution policy
  //     output->begin(),                         // input iterator: begin
  //     output->end()                            // input iterator: end
  // );

  // new_size = thrust::distance(output->begin(), new_end);
  // output->resize(new_size);
}
#else

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Resize frontier as necessary
  if (output->size() != input->size()) {
    output->resize(input->size());
  }

  // Define op
  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? op(i) : false;
  };

  // Init GPUs
  int orig_device = 0;
  cudaGetDevice(&orig_device);
  
  struct gpu_info {
    cudaStream_t stream;
    cudaEvent_t  event;
    type_t*      input_begin;
    type_t*      input_end;
    type_t*      output_begin;
  };
  
  std::vector<gpu_info> gpu_infos;

  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);
  auto chunk_size = (input->size() + num_gpus - 1) / num_gpus;
  
  for(int i = 0; i < num_gpus; i++) {
    gpu_info info;
    
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreate(&info.event);

    info.input_begin  = input->begin() + chunk_size * i;
    info.input_end    = input->begin() + chunk_size * (i + 1);
    info.output_begin = output->begin() + chunk_size * i;
    
    if(i == num_gpus - 1) info.input_end = input->end();
    
    gpu_infos.push_back(info);
  }

  // Run
  size_t new_sizes[num_gpus];
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    auto new_length = thrust::copy_if(
      thrust::cuda::par.on(gpu_infos[i].stream),
      gpu_infos[i].input_begin,
      gpu_infos[i].input_end,
      gpu_infos[i].output_begin,
      predicate
    );
    new_sizes[i] = thrust::distance(gpu_infos[i].output_begin, new_length);
    cudaEventRecord(gpu_infos[i].event, gpu_infos[i].stream);
  }

  // Sync
  for(int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(context.stream(), gpu_infos[i].event, 0);
  
  // Append results
  size_t offset = new_sizes[0];
  for(int i = 1; i < num_gpus; i++) {
    thrust::copy(
      thrust::device,
      gpu_infos[i].output_begin,
      gpu_infos[i].output_begin + new_sizes[i],
      gpu_infos[0].output_begin + offset
    );
    
    offset += new_sizes[i];
  }
  output->resize(offset);
  
  // Cleanup 
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaStreamDestroy(gpu_infos[i].stream);
    cudaEventDestroy(gpu_infos[i].event);
  }
  
  cudaSetDevice(orig_device);
}

#endif

}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock