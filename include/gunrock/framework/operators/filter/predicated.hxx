#pragma once

#include "omp.h"
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
  
  auto context0 = context.get_context(0);
  
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Resize frontier as necessary
  if (output->size() != input->size()) {
    output->resize(input->size());
  }

  // Define op
  auto fn = [=] __host__ __device__(type_t const& i) -> type_t {
    return gunrock::util::limits::is_valid(i) ? (op(i) ? i : -1) : -1;
  };
  
  // Setup
  int num_gpus    = context.size();
  auto chunk_size = (input->size() + num_gpus - 1) / num_gpus;
  
  // Map
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaSetDevice(ctx->ordinal());

    auto input_begin  = input->begin() + chunk_size * i;
    auto output_begin = output->begin() + chunk_size * i;
    auto input_end    = input->begin() + chunk_size * (i + 1);
    if(i == num_gpus - 1) input_end = input->end();

    thrust::transform(
      thrust::cuda::par.on(ctx->stream()),
      input_begin,
      input_end,
      output_begin,
      fn
    );
    cudaEventRecord(ctx->event(), ctx->stream());
  }
  
  // Sync
  for(int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(context0->stream(), context.get_context(i)->event(), 0);
  
  cudaSetDevice(context0->ordinal());
  auto new_end = thrust::remove(
    thrust::cuda::par.on(context0->stream()),
    output->begin(),
    output->end(),
    -1
  );
  auto new_size = thrust::distance(output->begin(), new_end);
  output->resize(new_size);
  
  cudaEventRecord(context0->event(), context0->stream());
  cudaStreamWaitEvent(context0->stream(), context0->event(), 0);
}

}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock