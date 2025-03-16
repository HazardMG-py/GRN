# methods/ledoit_wolf.py
import numpy as np
from sklearn.covariance import LedoitWolf

def compute_ledoit_wolf(G, threshold=0.01):
    """Infer sparse adjacency matrix using Ledoit-Wolf shrinkage."""
    lw = LedoitWolf()
    lw.fit(G.T)
    covariance = lw.covariance_
    precision_matrix = np.linalg.inv(covariance)
    A = (np.abs(precision_matrix) > threshold).astype(float)
    np.fill_diagonal(A, 0)
    return A