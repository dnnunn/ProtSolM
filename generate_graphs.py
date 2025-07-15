import argparse
import yaml
import os
from src.dataset.supervise_dataset import SuperviseDataset
from src.utils.common import set_seed


def create_parser():
    parser = argparse.ArgumentParser()
    # Add only the arguments needed for graph generation from eval.py
    parser.add_argument("--supv_dataset", type=str, default="data/PDBSol", help="Supervised dataset path")
    parser.add_argument("--test_file", type=str, required=True, help="Test CSV file with protein IDs")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the directory containing PDB files")
    parser.add_argument("--gnn", type=str, default="prossn", help="GNN model name")
    parser.add_argument("--gnn_config", type=str, default="./config/gnn.yml", help="GNN config file")
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=20, help="Graph dataset K")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_parser()
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    set_seed(args.seed)

    print("--- Initializing Dataset to Generate Protein Graphs ---")
    # We instantiate the dataset object, which will trigger graph generation.
    # We pass an empty feature_dict because features are not needed for graph creation.
    dataset = SuperviseDataset(
        args,
        root=args.supv_dataset,
        split='test',
        test_file=args.test_file,
        feature_dict={},
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pdb_path=args.pdb_path,
    )

    print(f"--- Graph generation complete. ---")
    print(f"Processed files should now be in: {dataset.processed_dir}")
