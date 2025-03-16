# data/preprocessing.py
import torch
from sklearn.preprocessing import StandardScaler

def preprocess_data(G, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Normalize gene expression data and convert to tensor."""
    scaler = StandardScaler()
    G_norm = scaler.fit_transform(G)
    G_tensor = torch.FloatTensor(G_norm).to(device)
    return G_tensor

def adjacency_to_edge_index(A, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Convert adjacency matrix to edge_index format."""
    edge_index = torch.tensor(np.array(np.where(A > 0)), dtype=torch.long).to(device)
    return edge_index