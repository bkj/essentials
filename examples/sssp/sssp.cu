#include <cstdlib>  // EXIT_SUCCESS

#include <examples/sssp/sssp_implementation.hxx>

using namespace gunrock;

void run_app() {
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  // --
  // Devices + Context

  std::vector<cuda::device_id_t> devices;
  devices.push_back(0);

  auto multi_context = std::shared_ptr<cuda::multi_context_t>(
      new cuda::multi_context_t(devices));

  // --
  // Params

  vertex_t single_source = 0;

  // --
  // Init CSR w/ dummy data

  std::ifstream prob("chesapeake.bin", std::ios::in | std::ios::binary);

  vertex_t n_nodes;
  edge_t n_edges;

  prob.read((char*)&n_nodes, sizeof(vertex_t));
  prob.read((char*)&n_edges, sizeof(edge_t));

  thrust::host_vector<edge_t> h_offsets(n_nodes + 1);
  thrust::host_vector<vertex_t> h_indices(n_edges);
  thrust::host_vector<weight_t> h_values(n_edges);

  prob.read((char*)h_offsets.data(), sizeof(edge_t)   * (n_nodes + 1));
  prob.read((char*)h_indices.data(), sizeof(vertex_t) * n_edges);
  prob.read((char*)h_values.data(),  sizeof(weight_t) * n_edges);

  thrust::copy(h_offsets.begin(), h_offsets.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  printf("\n");
  thrust::copy(h_indices.begin(), h_indices.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  printf("\n");
  thrust::copy(h_values.begin(), h_values.end(),   std::ostream_iterator<weight_t>(std::cout, " "));
  printf("\n");

  thrust::device_vector<edge_t>   d_offsets = h_offsets;
  thrust::device_vector<vertex_t> d_indices = h_indices;
  thrust::device_vector<weight_t> d_values  = h_values;

  // --
  // Init graphs

  auto h_G = graph::build::from_csr_t<memory::memory_space_t::host>(
    n_nodes, n_nodes, n_edges, d_offsets, d_indices, d_values
  ); // !! Illegal device memory -- why?

  auto d_G = graph::build::from_csr_t<memory::memory_space_t::device>(
    n_nodes, n_nodes, n_edges, d_offsets, d_indices, d_values
  );

  // --
  // Init problem

  using d_graph_type = decltype(d_G)::value_type;
  using h_graph_type = decltype(h_G)::value_type;
  using problem_type = sssp::sssp_problem_t<d_graph_type, h_graph_type>;
  using enactor_type = sssp::sssp_enactor_t<problem_type>;

  problem_type problem(
    d_G.data().get(), // Don't love having to call .data().get() here
    h_G.data(),       // Don't love having to call .data() here
    multi_context,
    single_source
  );

  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());

  // --
  // Init + Run enactor

  enactor_type enactor(&problem, multi_context);
  float elapsed = enactor.enact();

  // --
  // Print results

  std::cout << "Distances (output) = ";
  thrust::copy(problem.distances_vec.begin(), problem.distances_vec.end(),
    std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  run_app();
  return EXIT_SUCCESS;
}