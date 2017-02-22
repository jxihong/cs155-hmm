import numpy as np

import graphviz as gv

def create_graph(M):
    """
    Creates a weighted directed graph with transition matrix. Make
    sure add start probabilities as the first row, making the START
    state the 0th hidden state.
    """
    G = gv.Digraph(format='png')
    
    n, m = M.shape
    G.node('0', 'Start')
    G = add_nodes(G, map(str, range(1, n)))
    
    rows, cols = np.where(M >= 0.01)
    weights = M[rows, cols]
    
    rows = map(str, rows.tolist())
    cols = map(str, cols.tolist())
    
    edges = zip(rows, cols)
    w_edges = zip(edges, map(lambda x: "%0.2f" % (x), weights))
    
    G = add_edges(G, w_edges)
    
    return G


def add_nodes(graph, nodes):
    for n in nodes:
        graph.node(n)
    return graph
    

def add_edges(graph, edges):
    for e in edges:
        # for weighted edges
        if isinstance(e[0], tuple):
            graph.edge(*(e[0] + (e[1],)))
        else:
            graph.edge(e)
    return graph
    
    
if __name__=='__main__':
    import os

    # Create a graph from random matrix
    M = np.array([[3.3e-16, 0.33, 1.2e-4],
                  [0.5, 0.5, 0.],
                  [9.9999e-1, 0., 1e-120]])
    
    create_graph(M).render('test')

    os.remove('test') # Remove the dot file
