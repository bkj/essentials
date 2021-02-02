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

  int num_gpus    = context.size();
  auto chunk_size = (input->size() + num_gpus - 1) / num_gpus;

  type_t* input_begins[num_gpus];
  type_t* input_ends[num_gpus];
  type_t* output_begins[num_gpus];
  size_t new_sizes[num_gpus];
  
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = mcontext.get_context(i);
    cudaSetDevice(ctx._ordinal);
    
    auto new_length = thrust::copy_if(
      thrust::cuda::par.on(ctx.stream()),
      input_begins[i],
      input_ends[i],
      output_begins[i],
      predicate
    );
    new_sizes[i] = thrust::distance(output_begins[i], new_length);
    cudaEventRecord(ctx.event(), ctx.stream());
  }

  // Sync
  for(int i = 0; i < num_gpus; i++) {
    auto ctx = context.get_context(i);
    cudaStreamWaitEvent(context.get_context(0).stream(), ctx.event(), 0);
  }
  
  // Append results
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
  
  cudaSetDevice(mcontext.get_context(0)._ordinal);
}

#endif

}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock