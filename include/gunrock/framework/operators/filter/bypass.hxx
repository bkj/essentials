#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {

template <typename graph_t, typename operator_t, typename frontier_t>
void execute_gpu(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // ... resize as needed.
  if ((output->data() != input->data()) || (output->size() != input->size())) {
    output->resize(input->size());
  }

  // Mark items as invalid instead of removing them (therefore, a "bypass").
  auto bypass = [=] __device__(type_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<type_t>::invalid());
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
void execute_mgpu(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::multi_context_t& context) {
  
  using type_t = std::remove_pointer_t<decltype(input->data())>;
  
  // Resize frontier as necessary
  if ((output->data() != input->data()) || (output->size() != input->size())) {
    output->resize(input->size());
  }

  // Define op
  auto bypass = [=] __device__(type_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<type_t>::invalid());
  };
  
  // Setup
  int num_gpus    = context.size();
  auto chunk_size = (input->size() + num_gpus - 1) / num_gpus;

  type_t* input_begins[num_gpus];
  type_t* input_ends[num_gpus];
  type_t* output_begins[num_gpus];

  for(int i = 0; i < num_gpus; i++) {
    input_begins[i]  = input->begin() + chunk_size * i;
    input_ends[i]    = input->begin() + chunk_size * (i + 1);
    output_begins[i] = output->begin() + chunk_size * i;
  }
  input_ends[num_gpus - 1] = input->end();
  
  // Map
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaSetDevice(ctx->ordinal());
    
    thrust::transform(
      thrust::cuda::par.on(ctx->stream()),
      input_begins[i],
      input_ends[i],
      output_begins[i],
      bypass
    );
    
    cudaEventRecord(ctx->event(), ctx->stream());
  }

  // Sync
  auto context0 = context.get_context(0);
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaStreamWaitEvent(context0->stream(), ctx->event(), 0);
  }
  
  cudaSetDevice(context0->ordinal());
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute_gpu(graph_t& G,
             operator_t op,
             frontier_t* input,
             cuda::standard_context_t& context) {
  execute_gpu(G, op, input, input, context);
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute_mgpu(graph_t& G,
             operator_t op,
             frontier_t* input,
             cuda::multi_context_t& context) {
  execute_mgpu(G, op, input, input, context);
}

}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock