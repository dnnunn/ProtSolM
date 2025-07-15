import argparse
import yaml
import os
from src.dataset.supervise_dataset import SuperviseDataset
from src.utils.common import set_seed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_parser():
    parser = argparse.ArgumentParser(description="Pre-process PDB files into graph .pt files for ProtSolM evaluation.")
    parser.add_argument("--test_file", type=str, required=True, help="Test CSV file containing protein IDs (e.g., data/PDBSol/test_sequences.csv).")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the directory containing all your PDB files.")
    parser.add_argument("--supv_dataset_path", type=str, default="data/PDBSol", help="Root directory for the supervised dataset.")
    parser.add_argument("--gnn_config", type=str, default="./config/gnn.yml", help="GNN config file.")
    parser.add_argument("--gnn", type=str, default="prossn", help="GNN model name.")
    parser.add_argument("--k_neighbors", type=int, default=20, help="Number of neighbors (K) for graph construction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_parser()
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    # Manually add hidden_channels to args for SuperviseDataset compatibility
    args.gnn_hidden_dim = args.gnn_config.get('hidden_channels', 512)
    set_seed(args.seed)

    logging.info("--- Initializing Dataset to Generate Protein Graphs ---")
    logging.info(f"Reading PDBs from: {args.pdb_path}")
    logging.info(f"Reading protein list from: {args.test_file}")

    # This is the critical fix: We override the 'processed_dir' to match what eval.py expects.
    # This ensures the .pt files are saved in the correct location.
    class PatchedSuperviseDataset(SuperviseDataset):
        @property
        def processed_dir(self):
            # Construct the exact path that eval.py will look for.
            graph_dir_name = f"Pdbsol_k{self.c_alpha_max_neighbors}"
            path = os.path.join(self.root, graph_dir_name, "processed")
            os.makedirs(path, exist_ok=True)
            return path

    # Instantiate the patched dataset class. This will automatically trigger graph generation.
    dataset = PatchedSuperviseDataset(
        args,
        root=args.supv_dataset_path,
        split='test',
        test_file=args.test_file,
        feature_dict={}, # Features are not needed for graph creation
        c_alpha_max_neighbors=args.k_neighbors,
        pdb_path=args.pdb_path,
    )

    logging.info(f"--- Graph generation complete. ---")
    logging.info(f"Processed files saved in: {dataset.processed_dir}")
    logging.info("You can now run eval.py successfully.")
