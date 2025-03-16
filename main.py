# main.py
import argparse
import torch
from data.loader import load_dream5_ecoli
from data.preprocessing import preprocess_data, adjacency_to_edge_index
from methods.graphical_lasso import compute_graphical_lasso, compute_ledoit_wolf
from methods.vae import VAE, train_vae
from methods.cgat import CGAT, train_cgat, conformal_uncertainty
from methods.link_prediction import link_prediction
from evaluation.metrics import evaluate_predictions


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess data
    G, A_true, gene_ids = load_dream5_ecoli()
    G_tensor = preprocess_data(G, device)
    A_true_tensor = torch.FloatTensor(A_true).to(device)

    # Graphical Lasso with alpha from args
    #A = compute_graphical_lasso(G, alpha=args.gl_alpha)
    #edge_index = adjacency_to_edge_index(A, device)
    # Ledoit-Wolf instead of Graphical Lasso
    A = compute_ledoit_wolf(G, threshold=args.lw_threshold)
    edge_index = adjacency_to_edge_index(A, device)

    # VAE with hyperparameters from args
    vae = VAE(input_dim=G.shape[1], hidden_dim=args.vae_hidden_dim, latent_dim=args.vae_latent_dim).to(device)
    Z = train_vae(vae, G_tensor, epochs=args.vae_epochs, lr=args.vae_lr)

    # C-GAT with TE integration and hyperparameters from args
    cgat = CGAT(in_channels=args.vae_latent_dim, hidden_channels=args.cgat_hidden_channels,
                out_channels=args.cgat_out_channels, heads=args.cgat_heads).to(device)
    h = train_cgat(cgat, Z, edge_index, G_tensor, A_true_tensor, epochs=args.cgat_epochs, lr=args.cgat_lr)
    uncertainty = conformal_uncertainty(h, alpha=args.conformal_alpha)

    # Link Prediction
    A_hat_tensor = link_prediction(h)
    A_hat = A_hat_tensor.cpu().numpy()

    # Evaluation
    auroc, auprc, f1 = evaluate_predictions(A_true, A_hat)
    print(f"E. coli Results - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRN Inference with C-GAT and TE")

    # Graphical Lasso hyperparameters
    parser.add_argument("--gl_alpha", type=float, default=0.1, help="L1 regularization for Graphical Lasso")

    # VAE hyperparameters
    parser.add_argument("--vae_epochs", type=int, default=100, help="Number of epochs for VAE training")
    parser.add_argument("--vae_lr", type=float, default=0.001, help="Learning rate for VAE")
    parser.add_argument("--vae_hidden_dim", type=int, default=128, help="Hidden dimension for VAE")
    parser.add_argument("--vae_latent_dim", type=int, default=64, help="Latent dimension for VAE")

    # C-GAT hyperparameters
    parser.add_argument("--cgat_epochs", type=int, default=100, help="Number of epochs for C-GAT training")
    parser.add_argument("--cgat_lr", type=float, default=0.001, help="Learning rate for C-GAT")
    parser.add_argument("--cgat_hidden_channels", type=int, default=16, help="Hidden channels for C-GAT")
    parser.add_argument("--cgat_out_channels", type=int, default=32, help="Output channels for C-GAT")
    parser.add_argument("--cgat_heads", type=int, default=4, help="Number of attention heads for C-GAT")
    parser.add_argument("--conformal_alpha", type=float, default=0.1, help="Alpha for conformal prediction")

    args = parser.parse_args()
    main(args)