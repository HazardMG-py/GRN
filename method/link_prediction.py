# methods/link_prediction.py
import torch

def link_prediction(h):
    """Predict adjacency matrix using dot product similarity."""
    scores = h @ h.T
    return torch.sigmoid(scores)