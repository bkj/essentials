/**
 * @file build.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace graph {
namespace build {

/**
 * @brief
 *
 * @tparam graph_type
 * @param I
 * @return graph
 */
template <typename graph_type>
__device__ __host__ auto from_graph_t(graph_type& I) {
  graph_type G;
  G.set(I.get_number_of_rows(),      // r
        I.get_number_of_columns(),   // c
        I.get_number_of_nonzeros(),  // nnz
        I.get_row_offsets(),         // offsets
        I.get_column_indices(),      // column indices
        I.get_nonzero_values()       // nonzero values
  );

  return G;
}

template <typename graph_type>
__host__ __device__ void fix_virtual_inheritance(graph_type I, graph_type* O) {
  auto G = from_graph_t(I);
  memcpy(O, &G, sizeof(graph_type));
}

namespace device {
/**
 * @brief Instantiate polymorphic inhertance within the kernel & set the
 * existing data to it. No allocations allowed here.
 *
 * @tparam graph_type
 * @param I
 * @param O
 */
template <typename graph_type>
__global__ void kernel_virtual_inheritance(graph_type I, graph_type* O) {
  fix_virtual_inheritance(I, O);
}

/**
 * @brief Possible work around while keeping virtual (polymorphic behavior.)
 *
 * @tparam graph_type
 * @param r
 * @param c
 * @param nnz
 * @param Ap
 * @param Aj
 * @param Ax
 * @return auto
 */
template <typename graph_type>
void csr_t(graph_type I, graph_type* G) {
  kernel_virtual_inheritance<graph_type><<<1, 1>>>(I, G);
}
}  // namespace device

namespace host {
template <typename graph_type>
void csr_t(graph_type I, graph_type* G) {
  fix_virtual_inheritance(I, G);
}
}  // namespace host

template <memory_space_t space,
          typename edge_vector_t,
          typename vertex_vector_t,
          typename weight_vector_t>
auto from_csr_t(typename vertex_vector_t::value_type const& r,
                typename vertex_vector_t::value_type const& c,
                typename edge_vector_t::value_type const& nnz,
                edge_vector_t& Ap,
                vertex_vector_t& Aj,
                weight_vector_t& Ax) {
  using vertex_type = typename vertex_vector_t::value_type;
  using edge_type   = typename edge_vector_t::value_type;
  using weight_type = typename weight_vector_t::value_type;

  auto Ap_ptr = memory::raw_pointer_cast(Ap.data());
  auto Aj_ptr = memory::raw_pointer_cast(Aj.data());
  auto Ax_ptr = memory::raw_pointer_cast(Ax.data());

  using graph_type = graph::graph_t<
      space, vertex_type, edge_type, weight_type,
      graph::graph_csr_t<space, vertex_type, edge_type, weight_type>>;

  typename vector<graph_type, space>::type O(1);
  graph_type G;

  G.set(r, c, nnz, Ap_ptr, Aj_ptr, Ax_ptr);

  if (space == memory_space_t::device) {
    device::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));
  } else {
    host::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));
  }

  return O;
}

template <memory_space_t space, typename csr_t>
auto from_csr_t(csr_t* csr) {
  return from_csr_t<space>(
      csr->number_of_rows,      // number of rows
      csr->number_of_columns,   // number of columns
      csr->number_of_nonzeros,  // number of edges
      csr->row_offsets,         // row offsets
      csr->column_indices,      // column indices
      csr->nonzero_values);  
}

// template <typename vertex_t, typename edge_t>
// auto meta_graph(vertex_t const& r, vertex_t const& c, edge_t const& nnz) {
//   using vertex_type = vertex_t;
//   using edge_type   = edge_t;
//   using weight_type = edge_t;

//   constexpr memory_space_t space = memory_space_t::host;

//   using graph_type = graph::graph_t<
//       space, vertex_type, edge_type, weight_type,
//       graph::graph_csr_t<space, vertex_type, edge_type, weight_type>>;

//   typename vector<graph_type, space>::type O(1);
//   graph_type G;

//   G.set(r, c, nnz, nullptr, nullptr, nullptr);
//   host::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));

//   return O;
// }

template <typename csr_t>
auto meta_graph(csr_t* csr) {
  using vertex_type = typename decltype(csr->row_offsets)::value_type;
  using edge_type   = typename decltype(csr->column_indices)::value_type;
  using weight_type = typename decltype(csr->nonzero_values)::value_type;

  constexpr memory_space_t space = memory_space_t::host;

  using graph_type = graph::graph_t<
      space, vertex_type, edge_type, weight_type,
      graph::graph_csr_t<space, vertex_type, edge_type, weight_type>>;

  typename vector<graph_type, space>::type O(1);
  graph_type G;

  G.set(csr->number_of_rows, csr->number_of_columns, csr->number_of_nonzeros, nullptr, nullptr, nullptr);
  host::csr_t<graph_type>(G, memory::raw_pointer_cast(O.data()));

  return O;
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock