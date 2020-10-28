/**
 * @file sssp_implementation.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm. This is where
 * we actually implement SSSP using operators.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>
#include <bits/stdc++.h>

namespace gunrock {
namespace sssp {

template <typename meta_t>
struct sssp_param_t {
   using vertex_t = typename meta_t::vertex_type;
   
   vertex_t single_source;
   sssp_param_t(vertex_t single_source_) {
     single_source = single_source_;
   }
};

template <typename meta_t>
struct sssp_result_t {
  using vertex_t = typename meta_t::vertex_type;
  using weight_t = typename meta_t::weight_type;
   
  weight_t* distances;
  vertex_t* predecessors;
  
  sssp_result_t(
    weight_t* distances_,
    vertex_t* predecessors_
  ) {
    distances    = distances_;
    predecessors = predecessors_;
  }
};


template <typename d_graph_t, typename meta_t>
struct sssp_problem_t : problem_t<d_graph_t, meta_t> {

  using param_t   = sssp_param_t<meta_t>;
  using result_t  = sssp_result_t<meta_t>;

  using vertex_t = typename meta_t::vertex_type;
  using edge_t   = typename meta_t::edge_type;
  using weight_t = typename meta_t::weight_type;
  
  param_t* param;
  result_t* result;
  
  thrust::device_vector<vertex_t> visited;

  sssp_problem_t(
    d_graph_t* G,
    meta_t*    meta,
    param_t*   param_,
    result_t*  result_,
    std::shared_ptr<cuda::multi_context_t> context
  ) : problem_t<d_graph_t, meta_t>(G, meta, context), param(param_), result(result_) {
    
    // Set initial values
    // !! Ideally this owuld be in a `reset` function, but couldn't get it to work
    auto n_vertices = meta[0].get_number_of_vertices();
    
    auto d_distances = thrust::device_pointer_cast(result->distances);
    thrust::fill(
      thrust::device,
      d_distances + 0,
      d_distances + n_vertices,
      std::numeric_limits<weight_t>::max()
    );
    
    thrust::fill(
      thrust::device, 
      d_distances + param->single_source,
      d_distances + param->single_source + 1,
      0
    );
    
    visited.resize(n_vertices);
    thrust::fill(thrust::device, visited.begin(), visited.end(), -1);
  }

  sssp_problem_t(const sssp_problem_t& rhs) = delete;            // Boilerplate? Can remove?
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete; // Boilerplate? Can remove?
};

template <typename problem_t>
struct sssp_enactor_t : enactor_t<problem_t> {
  using enactor_type = enactor_t<problem_t>;
  
  using vertex_t = typename problem_t::vertex_t;
  using edge_t   = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  /**
   * @brief Populate the initial frontier with a single source node from where
   * we begin shortest path traversal.
   *
   * @param context
   */
  void prepare_frontier(cuda::standard_context_t* context) override {
    auto P = enactor_type::get_problem_pointer();
    auto f = enactor_type::get_active_frontier_buffer();
    f->push_back(P->param->single_source);
  }

  /**
   * @brief This is the core of the implementation for SSSP algorithm. loops
   * till the convergence condition is met (see: is_converged()). Note that this
   * function is on the host and is timed, so make sure you are writing the most
   * efficient implementation possible. Avoid performing copies in this function
   * or running API calls that are incredibly slow (such as printfs), unless
   * they are part of your algorithms' implementation.
   *
   * @param context
   */
  void loop(cuda::standard_context_t* context) override {
    // Data slice
    auto P = enactor_type::get_problem_pointer();
    auto G = P->get_graph_pointer();
    
    auto single_source = P->param->single_source;
    auto distances     = P->result->distances;
    auto visited       = P->visited.data().get();
    
    auto iteration = enactor_type::iteration;

    /**
     * @brief Lambda operator to advance to neighboring vertices from the
     * source vertices in the frontier, and marking the vertex to stay in the
     * frontier if and only if it finds a new shortest distance, otherwise,
     * it's shortest distance is found and we mark to remove the vertex from
     * the frontier.
     *
     */
    auto shortest_path = [distances, single_source] __host__ __device__(
      vertex_t const& source,    // ... source
      vertex_t const& neighbor,  // neighbor
      edge_t const& edge,        // edge
      weight_t const& weight     // weight (tuple).
    ) -> bool {
      weight_t source_distance = distances[source];  // use cached::load
      weight_t distance_to_neighbor = source_distance + weight;

      // Check if the destination node has been claimed as someone's child
      weight_t recover_distance =
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      return (distance_to_neighbor < recover_distance);
    };

    /**
     * @brief Lambda operator to determine which vertices to filter and which
     * to keep.
     *
     */
    auto remove_completed_paths = [visited, iteration] __host__ __device__(
                                      vertex_t const& vertex) -> bool {
      if (vertex == std::numeric_limits<vertex_t>::max())
        return false;
      if (visited[vertex] == iteration)
        return false;
      visited[vertex] = iteration;
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::merge_path>(
        G, enactor_type::get_enactor(), shortest_path, context);

    // Execute filter operator on the provided lambda
    operators::filter::execute<operators::filter_type_t::uniquify>(
        G, enactor_type::get_enactor(), remove_completed_paths, context);
  }

  sssp_enactor_t(problem_t* _problem,
                 std::shared_ptr<cuda::multi_context_t> _context)
      : enactor_type(_problem, _context) {}

  sssp_enactor_t(const sssp_enactor_t& rhs) = delete;            // Boilerplate? Can remove?
  sssp_enactor_t& operator=(const sssp_enactor_t& rhs) = delete; // Boilerplate? Can remove?
};  // struct sssp_enactor_t

// !! This should go somewhere else -- @neoblizz, where?
auto get_default_context() {
  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  return std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));
}

template <
  typename graph_vector_t,
  typename meta_vector_t,
  typename graph_t = typename graph_vector_t::value_type,
  typename meta_t  = typename meta_vector_t::value_type>
float run(
  graph_vector_t& G,
  meta_vector_t& meta,
  typename meta_t::vertex_type& single_source, // Parameter
  typename meta_t::weight_type* distances,     // Output
  typename meta_t::vertex_type* predecessors   // Output
) {
  
  using param_t   = gunrock::sssp::sssp_param_t<meta_t>;
  using result_t  = gunrock::sssp::sssp_result_t<meta_t>;
  using problem_t = gunrock::sssp::sssp_problem_t<graph_t, meta_t>;
  using enactor_t = gunrock::sssp::sssp_enactor_t<problem_t>;
  
  param_t  param(single_source);
  result_t result(distances, predecessors);

  // vv Shouldn't have to change anything below this line vv
  auto multi_context = get_default_context();

  problem_t problem(
      G.data().get(),    // input graph (GPU)
      meta.data(),       // metadata    (CPU)
      &param,            // input parameters
      &result,           // output results
      multi_context      // input context
  );

  enactor_t enactor(&problem, multi_context);

  return enactor.enact();
}

}  // namespace sssp
}  // namespace gunrock