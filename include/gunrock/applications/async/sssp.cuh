#pragma once

#include <bits/stdc++.h>
#include <gunrock/applications/application.hxx>

#include "gunrock/applications/async/queue.cuh"
#include "gunrock/applications/async/util/time.cuh"

namespace gunrock {
namespace async {
namespace sssp {

// <user-defined>
template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};
// </user-defined>

// <user-defined>
template <typename edge_t>
struct result_t {
  edge_t* distances;
  result_t(edge_t* _distances) : distances(_distances) {}
};
// </user-defined>

// This is very close to compatible w/ standard Gunrock problem_t
// However, it doesn't use the `context` argument, so not joining yet
template<typename graph_type, typename param_type, typename result_type>
struct problem_t {
  // <boiler-plate>
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;
  
  graph_type graph_slice;
  auto get_graph() {return graph_slice;}
  
  param_type param;
  result_type result;
  
  problem_t(
      graph_type&  G,
      param_type&  _param,
      result_type& _result
  ) : graph_slice(G), param(_param), result(_result) {}
  
  void init() {}
  // </boiler-plate>
  
  void reset() {
    // <user-defined>
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    
    auto single_source = param.single_source;
    auto d_distances       = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices, n_vertices + 1);
    thrust::fill(thrust::device, d_distances + single_source, d_distances + single_source + 1, 0);
    // </user-defined>
  }
};

// --
// Enactor

template<typename queue_t, typename val_t>
__global__ void _push_one(queue_t q, val_t val) {
    if(LANE_ == 0) q.push(val);
}

// This is very close to compatible w/ standard Gunrock enactor_t
// However, it doesn't use the `context` argument, so not joining yet
template<typename problem_t, typename single_queue_t=uint32_t>
struct enactor_t {
    using vertex_t = typename problem_t::vertex_t;
    using edge_t   = typename problem_t::edge_t;
    using weight_t = typename problem_t::weight_t;
    using queue_t  = MaxCountQueue::Queues<vertex_t, single_queue_t>;
    
    problem_t* problem;
    queue_t q;
    
    int numBlock  = 56 * 5;
    int numThread = 256;

    // <boiler-plate>
    enactor_t(
      problem_t* _problem,
      uint32_t  min_iter=800, 
      int       num_queue=4
    ) : problem(_problem) { 
        
        
        auto n_vertices = problem->get_graph().get_number_of_vertices();
        
        auto capacity = min(
          single_queue_t(1 << 30), 
          max(single_queue_t(1024),  single_queue_t(n_vertices * 1.5))
        );
        
        q.init(capacity, num_queue, min_iter);
        q.reset();
    }
    // <boiler-plate>

    // <user-defined>
    void prepare_frontier() {
      _push_one<<<1, 32>>>(q, problem->param.single_source);
    }
    // </user-defined>
    
    void enact() { // Is there some way to restructure this to follow the `loop` semantics?
      
      // <boiler-plate>
      prepare_frontier();
      // </boiler-plate>
      
      // <user-defined>
      auto G            = problem->get_graph();
      edge_t* distances = problem->result.distances;
      
      auto kernel = [G, distances] __device__ (vertex_t node, queue_t q) -> void {
          
          vertex_t node_distance = ((volatile vertex_t * )distances)[node];
          
          const vertex_t start  = G.get_starting_edge(node);
          const vertex_t degree = G.get_number_of_neighbors(node);
          
          for(int idx = 0; idx < degree; idx++) {
              vertex_t neib         = G.get_destination_vertex(start + idx);
              weight_t dist_to_neib = G.get_edge_weight(start + idx);
              vertex_t old_dist     = atomicMin(distances + neib, node_distance + dist_to_neib);
              if(old_dist > node_distance + dist_to_neib) {
                  q.push(neib);
              }
          }
      };
      // </user-defined>
      
      // <boiler-plate>
      q.launch_thread(numBlock, numThread, kernel);
      q.sync();
      // </boiler-plate>
  }
}; // struct enactor_t

template <typename graph_type>
float run(graph_type& G,
          typename graph_type::vertex_type& single_source,  // Parameter
          typename graph_type::edge_type* distances           // Output
) {
  
  // <user-defined>
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;

  using param_type   = param_t<vertex_t>;
  using result_type  = result_t<edge_t>;
  
  param_type param(single_source);
  result_type result(distances);
  // </user-defined>
  
  // <boiler-plate>
  using problem_type = problem_t<graph_type, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;
  
  problem_type problem(G, param, result);
  problem.init();
  problem.reset();
  
  enactor_type enactor(&problem);
  
  GpuTimer timer;
  timer.start();
  enactor.enact();
  timer.stop();
  return timer.elapsed();
  // </boiler-plate>
}

} // namespace sssp
} // namespace async
} // namespace gunrock