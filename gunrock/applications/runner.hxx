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
          typename d_graph_vector_t,
          typename h_graph_vector_t>
float graph_run(d_graph_vector_t& d_G,
           h_graph_vector_t& h_G,
           param_t& param,
           result_t& result) {

  // Create contexts for all the devices
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  std::shared_ptr<problem_t> problem(
    std::make_shared<problem_t>(
      d_G.data().get(),  // input graph (GPU)
      h_G.data(),        // input graph (CPU)
      multi_context,     // input context
      param,             // input source
      result));

  std::shared_ptr<enactor_t> enactor(
    std::make_shared<enactor_t>(
      problem.get(),  // pass in a problem (contains data in/out)
      multi_context));

  return enactor->enact();
}

template <typename problem_t, 
          typename enactor_t, 
          typename param_t,
          typename result_t,
          typename h_csr_t,
          typename d_csr_t>
float csr_run(h_csr_t h_csr,
              d_csr_t d_csr,
              param_t& param,
              result_t& result) {
  
  auto d_G = graph::build::from_csr_t<memory_space_t::device>(
      d_csr.number_of_rows,      // number of rows
      d_csr.number_of_columns,   // number of columns
      d_csr.number_of_nonzeros,  // number of edges
      d_csr.row_offsets,         // row offsets
      d_csr.column_indices,      // column indices
      d_csr.nonzero_values);     // nonzero values

  auto h_G = graph::build::from_csr_t<memory_space_t::host>(
      h_csr.number_of_rows,      // number of rows
      h_csr.number_of_columns,   // number of columns
      h_csr.number_of_nonzeros,  // number of edges
      h_csr.row_offsets,         // row offsets
      h_csr.column_indices,      // column indices
      h_csr.nonzero_values);     // nonzero values
  
  return graph_run<problem_t, enactor_t, param_t, result_t>(d_G, h_G, param, result);
}

}  // namespace gunrock