# methods/transfer_entropy.py
import numpy as np
import torch


def compute_te_matrix(G, k=1, threshold=0.05):
    """Compute TE matrix (simplified using correlation proxy)."""
    n_genes = G.shape[0]
    te_matrix = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:
                corr = np.corrcoef(G[i], G[j])[0, 1]
                te_matrix[i, j] = abs(corr) if abs(corr) > threshold else 0

    return torch.FloatTensor(te_matrix)

# For full TE: pip install pyinform
# from pyinform import transfer_entropy
# te_matrix[i, j] = transfer_entropy(G[j], G[i], k=k)