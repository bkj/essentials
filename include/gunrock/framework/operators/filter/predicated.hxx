#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace predicated {

template <typename graph_t, typename operator_t, typename frontier_t>
void execute_gpu(graph_t& G,
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
  output->resize(new_size);

  // Uniquify!
  // auto new_end = thrust::unique(
  //     thrust::cuda::par.on(context.stream()),  // execution policy
  //     output->begin(),                         // input iterator: begin
  //     output->end()                            // input iterator: end
  // );

  // new_size = thrust::distance(output->begin(), new_end);
  // output->resize(new_size);
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute_mgpu(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::multi_context_t& context) {
  
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Resize frontier as necessary
  if (output->size() != input->size()) {
    output->resize(input->size());
  }

  // Define op
  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? op(i) : false;
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
  size_t new_sizes[num_gpus];
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaSetDevice(ctx->ordinal());
    
    auto new_length = thrust::copy_if(
      thrust::cuda::par.on(ctx->stream()),
      input_begins[i],
      input_ends[i],
      output_begins[i],
      predicate
    );
    
    new_sizes[i] = thrust::distance(output_begins[i], new_length);
    cudaEventRecord(ctx->event(), ctx->stream());
  }
  
  // Sync
  auto context0 = context.get_context(0);
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaStreamWaitEvent(context0->stream(), ctx->event(), 0);
  }
  
  // Reduce
  size_t offset = new_sizes[0];
  for(int i = 1; i < num_gpus; i++) {
    thrust::copy(
      thrust::device,
      output_begins[i],
      output_begins[i] + new_sizes[i],
      output_begins[0] + offset
    );
    
    offset += new_sizes[i];
  }
  output->resize(offset);
  
  // Cleanup
  cudaSetDevice(context0->ordinal());
}

}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock