#!/usr/bin/env python

"""
    mtx2bin.py
"""

import sys
import numpy as np
from scipy.io.mmio import mmread

# --
# Filenames

inpath  = sys.argv[1]
assert '.mtx' in inpath
outpath = inpath.replace('.mtx', '.bin')

# --
# IO

inpath = 'chesapeake.mtx'

adj    = mmread(inpath).tocsr()

import networkx as nx
G    = nx.from_scipy_sparse_matrix(adj)
sssp = nx.single_source_shortest_path_length(G, 0)

[sssp.get(n, np.inf) for n in G.nodes]

len(list(nx.connected_components(G)))

# (adj @ adj.T).todense()

# --
# Pack problem + write

n_nodes = adj.shape[0]
n_edges = adj.nnz

prob = [
    np.array([n_nodes, n_edges]).astype(np.int32),
    adj.indptr.astype(np.int32),
    adj.indices.astype(np.int32),
    adj.data.astype(np.float32),
]
print(prob)

ba = bytearray(prob[0])
for xx in prob[1:]:
    ba.extend(bytearray(xx))

open(outpath, 'wb').write(ba)