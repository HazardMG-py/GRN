# data/loader.py
import pandas as pd
import numpy as np

def load_dream5_ecoli(data_dir="data/"):
    """Load E. coli DREAM5 expression and gold-standard data."""
    G_df = pd.read_csv(f"{data_dir}/net3_expression_data.tsv", sep="\t", index_col=0)
    G = G_df.values  # Shape: (4511, 805)

    gene_ids = pd.read_csv(f"{data_dir}/net3_gene_ids.tsv", sep="\t", header=None)[0].values
    G_df.index = gene_ids

    A_true_df = pd.read_csv(f"{data_dir}/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv",
                            sep="\t", header=None, names=["TF", "Target", "Weight"])
    A_true = np.zeros((len(gene_ids), len(gene_ids)))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_ids)}
    for _, row in A_true_df.iterrows():
        tf_idx = gene_to_idx[row["TF"]]
        target_idx = gene_to_idx[row["Target"]]
        A_true[tf_idx, target_idx] = row["Weight"]

    return G, A_true, gene_ids

if __name__ == "__main__":
    G, A_true, _ = load_dream5_ecoli()
    print(f"Expression shape: {G.shape}, Gold standard shape: {A_true.shape}")