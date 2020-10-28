#pragma once

#include <bits/stdc++.h>
#include <cstdlib>

#include <gunrock/applications/application.hxx>
#include <gunrock/algorithms/generate/random.hxx>

namespace gunrock {
namespace color {

template <typename meta_t>
struct color_param_t {
  // No parameters for this algorithm
};

template <typename meta_t>
struct color_result_t {
  using vertex_t = typename meta_t::vertex_type;
   
  vertex_t* colors;
  
  color_result_t(
    vertex_t* colors_
  ) {
    colors = colors_;
  }
};


template <typename graph_t, typename meta_t>
struct color_problem_t : problem_t<graph_t, meta_t> {

  using param_t   = color_param_t<meta_t>;
  using result_t  = color_result_t<meta_t>;

  using vertex_t = typename meta_t::vertex_type;
  using edge_t   = typename meta_t::edge_type;
  using weight_t = typename meta_t::weight_type;
  
  param_t* param;
  result_t* result;
  
  thrust::device_vector<vertex_t> randoms;

  color_problem_t(
    graph_t* G,
    meta_t*    meta,
    param_t*   param_,
    result_t*  result_,
    std::shared_ptr<cuda::multi_context_t> context
  ) : problem_t<graph_t, meta_t>(G, meta, context), param(param_), result(result_) {
    
    // XXX: Ugly. Initialize d_colors to be all INVALIDs.
    auto n_vertices = meta->get_number_of_vertices();
    auto d_colors   = thrust::device_pointer_cast(result->colors);
    thrust::fill(
      thrust::device,
      d_colors + 0,
      d_colors + n_vertices,
      std::numeric_limits<vertex_t>::max()
    );

    // Generate random numbers.
    algo::generate::random::uniform_distribution(0, n_vertices, randoms.begin());
  }

  color_problem_t(const color_problem_t& rhs) = delete;            // Boilerplate? Can remove?
  color_problem_t& operator=(const color_problem_t& rhs) = delete; // Boilerplate? Can remove?
};

template <typename problem_t>
struct color_enactor_t : enactor_t<problem_t> {
  using enactor_type = enactor_t<problem_t>;
  
  using vertex_t = typename problem_t::vertex_t;
  using edge_t   = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(cuda::standard_context_t* context) override {
    // Data slice
    auto E = enactor_type::get_enactor();
    auto P = E->get_problem_pointer();
    auto G = P->get_graph_pointer();

    auto colors    = P->result->colors;
    auto randoms   = P->randoms.data().get();
    auto iteration = E->iteration;

    auto color_me_in = [G, colors, randoms, iteration] __host__ __device__(
                           vertex_t const& vertex) -> bool {
      // If invalid vertex, exit early.
      if (vertex == std::numeric_limits<vertex_t>::max())
        return false;

      edge_t start_edge    = G->get_starting_edge(vertex);
      edge_t num_neighbors = G->get_number_of_neighbors(vertex);

      bool colormax = true;
      bool colormin = true;

      // Color two nodes at the same time.
      int color = iteration * 2;

      // Main loop that goes over all the neighbors and finds the maximum or
      // minimum random number vertex.
      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = G->get_destination_vertex(e);

        if ((colors[u] != std::numeric_limits<vertex_t>::max()) &&
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
    operators::filter::execute<operators::filter_type_t::uniquify>(
        G, E, color_me_in, context);
  }

  /**
   * @brief Populate the initial frontier with a the entire graph (nodes).
   *
   * @param context
   */
  void prepare_frontier(cuda::standard_context_t* context) override {
    auto E = enactor_type::get_enactor();      // Enactor pointer
    auto P = E->get_problem_pointer();         // Problem pointer
    auto g = P->get_host_graph_pointer();      // HOST graph pointer
    auto f = E->get_active_frontier_buffer();  // active frontier

    // XXX: Find a better way to initialize the frontier to all nodes
    for (vertex_t v = 0; v < g->get_number_of_vertices(); ++v)
      f->push_back(v);
  }

  color_enactor_t(problem_t* _problem,
                 std::shared_ptr<cuda::multi_context_t> _context)
      : enactor_type(_problem, _context) {}

  color_enactor_t(const color_enactor_t& rhs) = delete;            // Boilerplate? Can remove?
  color_enactor_t& operator=(const color_enactor_t& rhs) = delete; // Boilerplate? Can remove?
};  // struct color_enactor_t

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
  typename meta_t::vertex_type* colors   // Output
) {
  
  using param_t   = gunrock::color::color_param_t<meta_t>;
  using result_t  = gunrock::color::color_result_t<meta_t>;
  using problem_t = gunrock::color::color_problem_t<graph_t, meta_t>;
  using enactor_t = gunrock::color::color_enactor_t<problem_t>;
  
  param_t  param;
  result_t result(colors);

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

  // return enactor.enact();
  return -1;
}

}  // namespace color
}  // namespace gunrock