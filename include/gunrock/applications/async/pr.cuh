#pragma once

#include <bits/stdc++.h>
#include <gunrock/applications/application.hxx>

#include "gunrock/applications/async/queue.cuh"
#include "gunrock/applications/async/util/time.cuh"

namespace gunrock {
namespace async {
namespace pr {

// <user-defined>
template <typename weight_t>
struct param_t {
  weight_t lambda;
  weight_t epsilon;
  param_t(weight_t _lambda, weight_t _epsilon) : lambda(_lambda), epsilon(_epsilon) {}
};
// </user-defined>

// <user-defined>
template <typename weight_t>
struct result_t {
  weight_t* rank;
  weight_t* res;
  result_t(weight_t* _rank, weight_t* _res) : rank(_rank), res(_res) {}
};
// </user-defined>

template <typename graph_t, typename weight_t>
__global__ void _reset_rank_res(graph_t G, weight_t lambda, weight_t* rank, weight_t* res) {
    using vertex_t = typename graph_t::vertex_type;
    const vertex_t n_vertices = G.get_number_of_vertices();
    
    for(uint32_t src = TID; src < n_vertices; src += blockDim.x * gridDim.x) {
        rank[src] = 1.0 - lambda;
        
        const vertex_t start  = G.get_starting_edge(src);
        const vertex_t degree = G.get_number_of_neighbors(src);
        const weight_t update = (1.0 - lambda) * lambda / degree;
        
        for(uint32_t idx = 0; idx < degree; idx++) {
            const vertex_t dst = G.get_destination_vertex(start + idx);
            atomicAdd(res + dst, update);
        }
    }
}

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
    _reset_rank_res<<<320, 512>>>(
      this->get_graph(), 
      param.lambda, 
      this->result.rank, 
      this->result.res
    );
    // </user-defined>
  }
};

// --
// Enactor

template<typename queue_t, typename val_t>
__global__ void _push_one(queue_t q, val_t val) {
  if(LANE_ == 0) q.push(val);
}

template<typename queue_t, typename val_t>
__global__ void _push_all(queue_t q, val_t n) {
   for(val_t i = TID; i < n; i += blockDim.x * gridDim.x) q.push(i);
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

    // <boiler-plate<<<320, 512>>>>
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
      _push_all<<<numBlock, numThread>>>(q, problem->get_graph().get_number_of_vertices());
    }
    // </user-defined>
    
    void enact() { // Is there some way to restructure this to follow the `loop` semantics?
      
      // <boiler-plate>
      prepare_frontier();
      // </boiler-plate>
      
      // <user-defined>
      auto G           = problem->get_graph();
      weight_t lambda  = problem->param.lambda;
      weight_t epsilon = problem->param.epsilon;
      weight_t* rank   = problem->result.rank;
      weight_t* res    = problem->result.res;

      auto kernel = [G, lambda, epsilon, rank, res] __device__ (vertex_t node, queue_t q) -> void {
          // Yuxin's implementation had some kind of pre-fetching?
          // https://github.com/bkj/async-queue-paper/blob/cta/pr/pr.cuh#L213
          // Removing for simplicity -- I don't think it should change results
          
          weight_t res_owner         = atomicExch(res + node, 0.0);
          const vertex_t node_offset = G.get_starting_edge(node);
          const vertex_t degree      = G.get_number_of_neighbors(node);
          
          if(res_owner > 0) {
            atomicAdd(rank + node, res_owner);
            res_owner *= lambda / degree;
            
            for(vertex_t idx = 0; idx < degree; idx++) {
              const vertex_t neib     = G.get_destination_vertex(node_offset + idx);
              const weight_t old_rank = atomicAdd(res + neib, res_owner);
              if(old_rank <= epsilon && old_rank + res_owner >= epsilon)
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
          typename graph_type::weight_type& lambda,
          typename graph_type::weight_type& epsilon,
          typename graph_type::weight_type* rank,
          typename graph_type::weight_type* res
) {
  
  // <user-defined>
  using weight_t = typename graph_type::weight_type;

  using param_type   = param_t<weight_t>;
  using result_type  = result_t<weight_t>;
  
  param_type param(lambda, epsilon);
  result_type result(rank, res);
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

} // namespace bfs
} // namespace async
} // namespace gunrock