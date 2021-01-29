#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;

  // ... resize as needed.
  if ((output->data() != input->data()) || (output->size() != input->size())) {
    output->resize(input->size());
  }

  // Mark items as invalid instead of removing them (therefore, a "bypass").
  auto bypass = [=] __device__(vertex_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<vertex_t>::invalid());
  };

  // Filter with bypass
  thrust::transform(thrust::cuda::par.on(context.stream()),  // execution policy
                    input->begin(),   // input iterator: begin
                    input->end(),     // input iterator: end
                    output->begin(),  // output iterator
                    bypass            // predicate
  );
}

template <typename graph_t, typename operator_t, typename frontier_t>
void mgpu_execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  
  using vertex_t = typename graph_t::vertex_type;
  
  // Resize frontier as necessary
  if ((output->data() != input->data()) || (output->size() != input->size())) {
    output->resize(input->size());
  }

  // Define op
  auto bypass = [=] __device__(vertex_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<vertex_t>::invalid());
  };
  
  // Init GPUs
  int orig_device = 0;
  cudaGetDevice(&orig_device);
  
  struct gpu_info {
    cudaStream_t stream;
    cudaEvent_t  event;
    vertex_t*    input_begin;
    vertex_t*    input_end;
    vertex_t*    output_begin;
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
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    thrust::transform(
      thrust::cuda::par.on(gpu_infos[i].stream),
      gpu_infos[i].input_begin,
      gpu_infos[i].input_end,
      gpu_infos[i].output_begin,
      bypass
    );
    cudaEventRecord(gpu_infos[i].event, gpu_infos[i].stream);
  }

  // Sync
  for(int i = 0; i < num_gpus; i++) {
    cudaStreamWaitEvent(context.stream(), gpu_infos[i].event, 0);
  }
  
  // Cleanup 
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaStreamDestroy(gpu_infos[i].stream);
    cudaEventDestroy(gpu_infos[i].event);
  }
  
  cudaSetDevice(orig_device);
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             cuda::standard_context_t& context) {
  // in-place bypass filter (doesn't require an output frontier.)
  execute(G, op, input, input, context);
}

}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock