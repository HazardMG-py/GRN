# methods/cgat.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from methods.transfer_entropy import compute_te_matrix


class CGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(CGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.te_weights = None
        self.zeta = 0.001

    def set_te_weights(self, G, edge_index):
        te_matrix = compute_te_matrix(G.cpu().numpy())
        te_edge_weights = te_matrix[edge_index[0], edge_index[1]] + self.zeta
        self.te_weights = 1 / te_edge_weights.to(G.device)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        if self.te_weights is not None:
            h = h * self.te_weights.view(-1, 1)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        return h


def conformal_uncertainty(h, alpha=0.1):
    scores = torch.norm(h, dim=1)
    quantile = torch.quantile(scores, 1 - alpha)
    return scores > quantile


def train_cgat(cgat, Z, edge_index, G, A_true_tensor, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(cgat.parameters(), lr=lr)
    cgat.set_te_weights(G, edge_index)

    for epoch in range(epochs):
        h = cgat(Z, edge_index)
        A_hat = link_prediction(h)
        loss = F.binary_cross_entropy(A_hat, A_true_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"C-GAT Epoch {epoch}, Loss: {loss.item()}")
    return h