/**
 * @file sssp.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/applications/application.hxx>

#pragma once

namespace gunrock {
  
using namespace memory;
  
template <typename problem_t, 
          typename enactor_t, 
          typename param_t,
          typename result_t,
          typename graph_vector_t,
          typename meta_vector_t>
float run(graph_vector_t& G,
           meta_vector_t& meta,
           param_t& param,
           result_t& result) {

  // Create contexts for all the devices
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  std::shared_ptr<problem_t> problem(
    std::make_shared<problem_t>(
      G.data().get(),    // input graph (GPU)
      meta.data(),       // metadata    (CPU)
      multi_context,     // input context
      param,             // input parameters
      result));          // output results

  std::shared_ptr<enactor_t> enactor(
    std::make_shared<enactor_t>(
      problem.get(),
      multi_context));

  return enactor->enact();
}

}