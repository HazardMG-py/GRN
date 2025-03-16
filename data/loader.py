# data/loader.py
import pandas as pd
import numpy as np


def load_dream5_ecoli(data_dir="data/"):
    """Load E. coli DREAM5 expression and gold-standard data."""
    # Load expression data (expecting 4511 genes x 805 samples)
    G_df = pd.read_csv(f"{data_dir}/net3_expression_data.tsv", sep="\t", header=0, low_memory=False)
    G = G_df.values.astype(float)  # Shape: (805, 4511), header skipped
    print(f"Raw G shape: {G.shape}")

    # Transpose to (genes x samples)
    G = G.T  # Shape: (4511, 805)
    print(f"Adjusted G shape: {G.shape}")

    # Load gene IDs, treating first row as header
    gene_ids_df = pd.read_csv(f"{data_dir}/net3_gene_ids.tsv", sep="\t", header=0)
    gene_ids = gene_ids_df.iloc[:, 0].values  # Take first column, flatten to 1D
    print(f"Gene IDs shape: {gene_ids.shape}")

    # Check gene count
    num_genes = len(gene_ids)
    if num_genes not in [4511, 4512]:
        raise ValueError(f"Unexpected number of genes: {num_genes}. Expected 4511 or 4512.")

    # Handle mismatch between G and gene_ids
    if G.shape[0] != num_genes:
        print(f"Warning: Mismatch detected. G has {G.shape[0]} genes, gene_ids has {num_genes}.")
        if num_genes == 4512 and G.shape[0] == 4511:
            print("Trimming gene_ids to match G (4511 genes).")
            gene_ids = gene_ids[:4511]  # Take first 4511 to match G
            num_genes = 4511
        else:
            raise ValueError(f"Cannot resolve mismatch: G has {G.shape[0]} genes, gene_ids has {num_genes}.")

    # Validate sample count
    if G.shape[1] != 805:
        raise ValueError(f"Expected 805 samples, got {G.shape[1]}.")

    # Load gold standard
    A_true_df = pd.read_csv(f"{data_dir}/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv",
                            sep="\t", header=None, names=["TF", "Target", "Weight"])
    print(f"Gold standard raw shape: {A_true_df.shape}")
    assert A_true_df.shape[1] == 3, f"Expected 3 columns in gold standard, got {A_true_df.shape[1]}"

    # Build adjacency matrix
    A_true = np.zeros((num_genes, num_genes))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_ids)}
    for _, row in A_true_df.iterrows():
        tf_idx = gene_to_idx[row["TF"]]
        target_idx = gene_to_idx[row["Target"]]
        A_true[tf_idx, target_idx] = row["Weight"]

    return G, A_true, gene_ids


if __name__ == "__main__":
    G, A_true, gene_ids = load_dream5_ecoli()
    print(f"Expression shape: {G.shape}, Gold standard shape: {A_true.shape}, Gene IDs: {len(gene_ids)}")