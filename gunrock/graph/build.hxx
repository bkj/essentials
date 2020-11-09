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

#include <tuple>

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
          typename edge_type,
          typename vertex_type,
          typename weight_type>
auto from_csr_t(vertex_type const& r,
                vertex_type const& c,
                edge_type const& nnz,
                edge_type* Ap_ptr,
                vertex_type* Aj_ptr,
                weight_type* Ax_ptr) {
  
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

template <typename vertex_type, typename edge_type, typename weight_type>
auto meta_t(vertex_type const& r,
            vertex_type const& c,
            edge_type const& nnz) {

  constexpr memory_space_t h_space = memory_space_t::host;

  using meta_type = graph::graph_t<
      h_space, vertex_type, edge_type, weight_type,
      graph::graph_csr_t<h_space, vertex_type, edge_type, weight_type>>;

  typename vector<meta_type, h_space>::type P(1);
  meta_type M;

  M.set(r, c, nnz, nullptr, nullptr, nullptr);
  host::csr_t<meta_type>(M, memory::raw_pointer_cast(P.data()));
  
  return P;
}

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
  // From thrust vectors
  return from_csr_t<space>(
    r, c, nnz,
    memory::raw_pointer_cast(Ap.data()),
    memory::raw_pointer_cast(Aj.data()),
    memory::raw_pointer_cast(Ax.data())
  );
}

template <memory_space_t space, typename csr_t>
auto from_csr_t(csr_t* csr) {
  // From a CSR object
  return from_csr_t<space>(
      csr->number_of_rows,      // number of rows
      csr->number_of_columns,   // number of columns
      csr->number_of_nonzeros,  // number of edges
      csr->row_offsets,         // row offsets
      csr->column_indices,      // column indices
      csr->nonzero_values
  );  
}

}  // namespace build
}  // namespace graph
}  // namespace gunrock