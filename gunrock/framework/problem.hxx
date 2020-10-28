/**
 * @file problem.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/graph/graph.hxx>

namespace gunrock {
/**
 * @brief Inherit problem class for your custom applications' implementation.
 * Problem describes the data slice of your aglorithm, and the data can often be
 * replicated or partitioned to multiple instances (for example, in a multi-gpu
 * context). In the algorithms' problem constructor, initialize your data.
 *
 * @tparam graph_type
 * @tparam meta_type
 */
template <typename graph_type, typename meta_type>
struct problem_t {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  using vertex_pointer_t = typename graph_type::vertex_pointer_t;
  using edge_pointer_t = typename graph_type::edge_pointer_t;
  using weight_pointer_t = typename graph_type::weight_pointer_t;

  graph_type* graph_slice;
  meta_type* meta_slice;
  std::shared_ptr<cuda::multi_context_t> context;

  graph_type* get_graph_pointer() const { return graph_slice; }
  meta_type* get_meta_pointer() const { return meta_slice; }

  problem_t() : graph_slice(nullptr) {}
  problem_t(graph_type* G,
            meta_type* meta,
            std::shared_ptr<cuda::multi_context_t> _context)
      : graph_slice(G), meta_slice(meta), context(_context) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  problem_t(const problem_t& rhs) = delete;             // Copy constructor
  problem_t& operator=(const problem_t& rhs) = delete;  // Copy assignment

};  // struct problem_t

}  // namespace gunrock