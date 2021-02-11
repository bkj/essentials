/**
 * @file color.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Graph Coloring algorithm.
 * @version 0.1
 * @date 2020-11-24
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <bits/stdc++.h>
#include <cstdlib>
#include <cmath>

#include <gunrock/applications/application.hxx>
#include <gunrock/algorithms/generate/random.hxx>

namespace gunrock {
namespace color {

struct param_t {
  // No parameters for this algorithm
};

template <typename vertex_t>
struct result_t {
  vertex_t* colors;
  result_t(vertex_t* colors_) : colors(colors_) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;
  
  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  // int num_gpu = 4;
  // cudaGetDeviceCount(&num_gpu);
  
  edge_t* all_offsets[4];
  vertex_t* all_indices[4];
  
  edge_t* offsets;
  vertex_t* indices;

  thrust::device_vector<vertex_t> randoms;

  void reset() {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    auto n_edges = g.get_number_of_edges();
    auto d_colors = thrust::device_pointer_cast(this->result.colors);
    thrust::fill(thrust::device, d_colors + 0, d_colors + n_vertices,
                 gunrock::numeric_limits<vertex_t>::invalid());

    // Generate random numbers.
    randoms.resize(n_vertices);
    algo::generate::random::uniform_distribution(0, n_vertices,
                                                 randoms.begin());
    
    offsets = g.offsets;
    indices = g.indices;
    
    int num_gpu = 4;
    for(int i = 0; i < num_gpu; i++) {
      cudaMalloc((void **)&(all_offsets[i]), (n_vertices + 1) * sizeof(edge_t));
      cudaMemcpy(all_offsets[i], g.offsets, (n_vertices + 1) * sizeof(edge_t), cudaMemcpyDeviceToDevice);

      cudaMalloc((void **)&(all_indices[i]), n_edges * sizeof(vertex_t));
      cudaMemcpy(all_indices[i], g.indices, n_edges * sizeof(vertex_t), cudaMemcpyDeviceToDevice);
    }
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  // <user-defined>
  void prepare_frontier(cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto f = this->get_input_frontier();

    auto n_vertices = P->get_graph().get_number_of_vertices();

    // XXX: Find a better way to initialize the frontier to all nodes
    for (vertex_t v = 0; v < n_vertices; ++v)
      f->push_back(v);
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto colors = P->result.colors;
    auto randoms = P->randoms.data().get();
    auto iteration = E->iteration;

    auto all_offsets = P->all_offsets;
    auto all_indices = P->all_indices;

    auto offsets0 = all_offsets[0];
    auto indices0 = all_indices[0];
    auto offsets1 = all_offsets[1];
    auto indices1 = all_indices[1];
    auto offsets2 = all_offsets[2];
    auto indices2 = all_indices[2];
    auto offsets3 = all_offsets[3];
    auto indices3 = all_indices[3];

    auto color_me_in = [
      offsets0, indices0, offsets1, indices1, offsets2, indices2, offsets3, indices3,
      colors, randoms, iteration] __host__ __device__(
                           vertex_t const& vertex) -> bool {
      
      // int device = 0;
      // cudaGetDevice(&device);
      // printf("device=%d | vertex=%d\n", device, vertex);
      
      auto offsets = offsets0;
      auto indices = indices0;
      
      // <<
      // edge_t start_edge = G.get_starting_edge(vertex);
      // edge_t num_neighbors = G.get_number_of_neighbors(vertex);
      // --
      edge_t start_edge    = offsets[vertex];
      edge_t num_neighbors = offsets[vertex + 1] - offsets[vertex];
      // >>

      bool colormax = true;
      bool colormin = true;
      
      // Color two nodes at the same time.
      int color = iteration * 2;
      
      // Main loop that goes over all the neighbors and finds the maximum or
      // minimum random number vertex.
      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        // <<
        // vertex_t u = G.get_destination_vertex(e);
        // --
        vertex_t u = indices[e];
        // >>
      
        if (gunrock::util::limits::is_valid(colors[u]) &&
                (colors[u] != color + 1) && (colors[u] != color + 2) ||
            (vertex == u))
          continue;
        if (randoms[vertex] <= randoms[u])
          colormax = false;
        if (randoms[vertex] >= randoms[u])
          colormin = false;
      }

      // Color if the node has the maximum OR minimum random number, this way,
      // per iteration we can possibly fill 2 colors at the same time.
      if (colormax) {
        colors[vertex] = color + 1;
        return false;  // remove (colored).
      } else if (colormin) {
        colors[vertex] = color + 2;
        return false;  // remove (colored).
      } else {
        return true;  // keep (not colored).
      }
    };

    // Execute filter operator on the provided lambda.
    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, color_me_in, context);
  }
  // </user-defined>
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type* colors  // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;

  using param_type = param_t;
  using result_type = result_t<vertex_t>;

  param_type param;
  result_type result(colors);
  // </user-defined>

  // <boiler-plate>

  // >>
  int num_gpu = 0;
  cudaGetDeviceCount(&num_gpu);
  
  thrust::host_vector<int> devices;
  for(int device = 0 ; device < num_gpu; device++)
    devices.push_back(device);
  
  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(devices));
  
  multi_context->enable_peer_access();
  // --
  // auto multi_context =
  //     std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));
  // <<
  
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace color
}  // namespace gunrock