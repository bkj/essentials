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

template <typename graph_type, typename host_graph_type>
struct sssp_problem_t : problem_t<graph_type, host_graph_type> {

  using vertex_t = typename graph_type::vertex_type;
  using weight_t = typename graph_type::weight_type;

  vertex_t single_source;
  thrust::device_vector<weight_t> distances_vec;
  weight_t* distances;

  sssp_problem_t(graph_type* G,
                 host_graph_type* g,
                 std::shared_ptr<cuda::multi_context_t> context,
                 vertex_t& single_source_)
      : problem_t<graph_type, host_graph_type>(G, g, context) {

          // Store source
          single_source = single_source_;

          // Allocate + init distances
          distances_vec.resize(g->get_number_of_vertices());
          thrust::fill(thrust::device, distances_vec.begin(), distances_vec.end(), 100);
          thrust::fill(thrust::device, distances_vec.begin() + single_source, distances_vec.begin() + single_source + 1, 0);
          distances = distances_vec.data().get();
        }

  sssp_problem_t(const sssp_problem_t& rhs) = delete;            // Boilerplate -- can remove?
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete; // Boilerplate -- can remove?
};

template <typename algorithm_problem_t>
struct sssp_enactor_t : enactor_t<algorithm_problem_t> {
  using enactor_type = enactor_t<algorithm_problem_t>;
  using vertex_t     = typename algorithm_problem_t::vertex_t;
  using edge_t       = typename algorithm_problem_t::edge_t;
  using weight_t     = typename algorithm_problem_t::weight_t;

  void prepare_frontier(cuda::standard_context_t* context) override {
    auto P = enactor_type::get_problem_pointer();
    enactor_type::get_active_frontier_buffer()->push_back(P->single_source);
  }

  void loop(cuda::standard_context_t* context) override {
    auto P             = enactor_type::get_problem_pointer();
    auto distances     = P->distances;
    auto single_source = P->single_source;


    auto shortest_path = [distances, single_source] __host__ __device__(
      vertex_t const& source,    // ... source
      vertex_t const& neighbor,  // neighbor
      edge_t const& edge,        // edge
      weight_t const& weight     // weight (tuple).
    ) -> bool {
      weight_t source_distance      = distances[source];  // use cached::load
      weight_t distance_to_neighbor = source_distance + weight;

      weight_t recover_distance =
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      return (distance_to_neighbor < recover_distance);
    };

    // auto remove_completed_paths = [] __host__ __device__(
    //   vertex_t const& vertex
    // ) -> bool {

    //   return vertex != std::numeric_limits<vertex_t>::max();

    //   // !! I think this is still wrong... I think it's supposed to be
    //   //   distances[vertex] != std::numeric_limits<weight_t>::max()
    //   // but I'm still a little unclear on what this filter does anyway ..
    // };

    // --
    // Run

    auto G = P->get_graph_pointer();
    auto E = enactor_type::get_enactor();

    operators::advance::execute<operators::advance_type_t::vertex_to_vertex>(
        G, E, shortest_path);
    // operators::filter::execute<operators::filter_type_t::predicated>(
    //     G, E, remove_completed_paths);
  }

  sssp_enactor_t(algorithm_problem_t* problem,
                 std::shared_ptr<cuda::multi_context_t> context)
      : enactor_type(problem, context) {} // Boilerplate -- can remove?

  sssp_enactor_t(const sssp_enactor_t& rhs) = delete;             // Boilerplate -- can remove?
  sssp_enactor_t& operator=(const sssp_enactor_t& rhs) = delete;  // Boilerplate -- can remove?
};  // struct sssp_enactor_t

}  // namespace sssp
}  // namespace gunrock