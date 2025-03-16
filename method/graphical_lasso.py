# methods/graphical_lasso.py
import numpy as np
from sklearn.covariance import GraphicalLasso

def compute_graphical_lasso(G, alpha=0.1, threshold=0.01):
    """Infer sparse adjacency matrix using Graphical Lasso."""
    gl = GraphicalLasso(alpha=alpha)
    gl.fit(G.T)
    precision_matrix = gl.precision_
    A = (np.abs(precision_matrix) > threshold).astype(float)
    np.fill_diagonal(A, 0)
    return A