import os
import sys
import torch
import pandas as pd
import yaml
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from accelerate import Accelerator
from src.models import ProtssnClassification, PLM_model, GNN_model
from src.utils.data_utils import BatchSampler
from src.utils.utils import total_param_num, param_num
from src.dataset.supervise_dataset import SuperviseDataset
from src.utils.dataset_utils import NormalizeProtein

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("protsolm_predict_to_csv")

def process_data(name, feature_dict, label_dict, seq_dict, args, graph_dir):
    graph_name = name + '_final'
    pdb_name = name + '_final.pdb'
    data = torch.load(f"{args.supv_dataset}/{graph_dir.capitalize()}/processed/{graph_name}.pt")
    data.label = torch.tensor(label_dict[name]).view(1)
    data.aa_seq = seq_dict[name]
    data.name = name
    if pdb_name in feature_dict:
        feature_tensor = torch.from_numpy(feature_dict[pdb_name]).float().unsqueeze(0)
        logger.info(f"Loaded features for '{pdb_name}' with sum: {torch.sum(feature_tensor)}")
        data.feature = feature_tensor
    else:
        logger.info(f"No features found for '{pdb_name}', using default zeros")
        data.feature = torch.zeros(1, args.feature_dim)
    return data

def collect_fn(batch, feature_dict, label_dict, seq_dict, args, graph_dir):
    batch_data = []
    from copy import deepcopy
    from concurrent.futures import ThreadPoolExecutor, as_completed
    feature_d = deepcopy(feature_dict)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_data, name, feature_d, label_dict, seq_dict, args, graph_dir) for name in batch]
        for future in as_completed(futures):
            graph = future.result()
            batch_data.append(graph)
    return batch_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ProtSolM batch prediction to CSV (standardized output)")
    parser.add_argument('--supv_dataset', required=True, help='Path to supervised dataset root')
    parser.add_argument('--test_file', required=True, help='CSV with columns: name, aa_seq, label')
    parser.add_argument('--feature_file', required=True, help='CSV with handcrafted features')
    parser.add_argument('--gnn_model_path', required=True, help='Path to trained GNN model checkpoint')
    parser.add_argument('--gnn', required=True, help='GNN type')
    parser.add_argument('--gnn_hidden_dim', type=int, required=True)
    parser.add_argument('--plm', required=True, help='Path or ID for ESM model')
    parser.add_argument('--feature_name', default='biotite')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--pooling_method', default='attention1d')
    parser.add_argument('--plm_hidden_size', type=int, default=1280)
    parser.add_argument('--pooling_dropout', type=float, default=0.1)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--batch_token_num', type=int, default=4096)
    parser.add_argument('--max_graph_token_num', type=int, default=1024)
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=32)
    parser.add_argument('--output_csv', required=True, help='Output CSV path for standardized predictions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gnn_config', type=str, default='src/config/egnn.yaml', help='Path to GNN config YAML (default matches eval.py)')
    parser.add_argument('--feature_embed_dim', type=int, default=None, help='Feature embedding dimension (default: None, matches eval.py)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load features (EXACT logic from eval.py for full compatibility)
    logger.info("***** Loading Feature *****")
    feature_df = pd.read_csv(args.feature_file).set_index('protein name')
    if 'L' in feature_df.columns:
        feature_df = feature_df.drop('L', axis=1)
    args.feature_dim = 0
    feature_dict = {}
    # Define feature sets as in eval.py
    feature_aa_composition = [
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
    ]
    feature_gravy = ["gravy"]
    feature_ss_composition = ["ss8-G", "ss8-H", "ss8-I", "ss8-B", "ss8-E", "ss8-T", "ss8-S", "ss8-P", "ss8-L", "ss3-H", "ss3-E", "ss3-C"]
    feature_hygrogen_bonds = ["Hydrogen bonds", "Hydrogen bonds per 100 residues"]
    feature_exposed_res_fraction = [
        "Exposed residues fraction by 0%", "Exposed residues fraction by 5%", "Exposed residues fraction by 10%", 
        "Exposed residues fraction by 15%", "Exposed residues fraction by 20%", "Exposed residues fraction by 25%", 
        "Exposed residues fraction by 30%", "Exposed residues fraction by 35%", "Exposed residues fraction by 40%", 
        "Exposed residues fraction by 45%", "Exposed residues fraction by 50%", "Exposed residues fraction by 55%", 
        "Exposed residues fraction by 60%", "Exposed residues fraction by 65%", "Exposed residues fraction by 70%", 
        "Exposed residues fraction by 75%", "Exposed residues fraction by 80%", "Exposed residues fraction by 85%", 
        "Exposed residues fraction by 90%", "Exposed residues fraction by 95%", "Exposed residues fraction by 100%"
    ]
    feature_pLDDT = ["pLDDT"]
    # Parse feature_name as list
    if isinstance(args.feature_name, str):
        if args.feature_name.startswith("[") and args.feature_name.endswith("]"):
            import ast
            args.feature_name = ast.literal_eval(args.feature_name)
        else:
            args.feature_name = [args.feature_name]
    # Prepare feature sub-frames if needed
    feature_dict = {}
    for i in range(len(feature_df)):
        name = feature_df.index[i].split(".")[0]
        feature_dict[name] = []
        for feature_set in args.feature_name:
            if feature_set.lower() == "aa_composition":
                feature_cols = [col for col in feature_aa_composition if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'aa_composition' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            elif feature_set.lower() == "gravy":
                feature_cols = [col for col in feature_gravy if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'gravy' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            elif feature_set.lower() == "ss_composition":
                feature_cols = [col for col in feature_ss_composition if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'ss_composition' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            elif feature_set.lower() == "hygrogen_bonds":
                feature_cols = [col for col in feature_hygrogen_bonds if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'hygrogen_bonds' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            elif feature_set.lower() == "exposed_res_fraction":
                feature_cols = [col for col in feature_exposed_res_fraction if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'exposed_res_fraction' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            elif feature_set.lower() == "plddt":
                feature_cols = [col for col in feature_pLDDT if col in feature_df.columns]
                if not feature_cols:
                    logger.warning(f"No columns found for feature set 'pLDDT' in feature file. Skipping...")
                    continue
                feature_dict[name] += list(feature_df[feature_cols].iloc[i])
                args.feature_dim += len(feature_cols)
            else:
                logger.warning(f"Unknown feature set '{feature_set}' in feature_name. Skipping...")
    # If no feature_name matches, fallback to all columns (legacy biotite)
    if args.feature_dim == 0:
        feature_columns = feature_df.columns
        args.feature_dim = len(feature_columns)
        for pdb_name, row in feature_df.iterrows():
            feature_dict[pdb_name.split(".")[0]] = row.values.astype(float)
    # === Robust feature dimension check ===
    EXPECTED_FEATURE_DIM = 1792  # This must match the training/checkpoint expectation
    if args.feature_dim != EXPECTED_FEATURE_DIM:
        logger.error(f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, but got {args.feature_dim}.\n"
                     f"Check your feature file and ensure it matches the training feature set exactly in columns, order, and count.\n"
                     f"The checkpoint will not load unless this matches.\n"
                     f"If you do not have the correct feature file, you must obtain it from training or reconstruct it.")
        sys.exit(1)
    logger.info("***** Loading Dataset *****")
    test_df = pd.read_csv(args.test_file)
    test_names = test_df['id'].tolist()
    label_dict = dict(zip(test_df['id'], test_df['label']))
    seq_dict = dict(zip(test_df['id'], test_df['aa_seq']))
    graph_dir = 'esmfold_pdb' if os.path.exists(f"{args.supv_dataset}/esmfold_pdb") else 'esmfold_pdb'
    test_node_nums = [1 for _ in test_names]  # Placeholder: update if needed

    # Build dataloader
    test_dataloader = DataLoader(
        dataset=test_names, num_workers=4, 
        collate_fn=lambda x: collect_fn(x, feature_dict, label_dict, seq_dict, args, graph_dir),
        batch_sampler=BatchSampler(
            node_num=test_node_nums,
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=False
        )
    )

    # Load models
    logger.info("***** Load Model *****")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Restore YAML config loading for full compatibility
    import yaml
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    args.gnn_config["hidden_channels"] = args.gnn_hidden_dim  # Match eval.py behavior
    plm_model = PLM_model(args).to(device)
    gnn_model = GNN_model(args).to(device)
    checkpoint = torch.load(args.gnn_model_path)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not (
        'batch_norm1.running_mean' in k or
        'batch_norm1.running_var' in k or
        'batch_norm1.num_batches_tracked' in k or
        'batch_norm2.running_mean' in k or
        'batch_norm2.running_var' in k or
        'batch_norm2.num_batches_tracked' in k
    )}
    protssn_classification = ProtssnClassification(args)
    protssn_classification.to(device)
    protssn_classification.load_state_dict(filtered_state_dict, strict=False)
    for param in plm_model.parameters():
        param.requires_grad = False
    for param in gnn_model.parameters():
        param.requires_grad = False
    accelerator = Accelerator()
    protssn_classification, test_dataloader = accelerator.prepare(
        protssn_classification, test_dataloader
    )
    protssn_classification.eval()

    # Run prediction
    logger.info("***** Running prediction *****")
    result_dict = {"id": [], "aa_seq": [], "label": [], "pred_label": [], "probability": []}
    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            logits, _ = protssn_classification(plm_model, gnn_model, batch, True)
            logits = logits.cuda()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred_labels = torch.argmax(logits, 1).cpu().numpy()
            result_dict["id"].extend([data.name for data in batch])
            result_dict["aa_seq"].extend([data.aa_seq for data in batch])
            result_dict["label"].extend([data.label.item() for data in batch])
            result_dict["pred_label"].extend(pred_labels.tolist())
            result_dict["probability"].extend(probs.tolist())

    # Save standardized CSV
    df = pd.DataFrame({
        "id": result_dict["id"],
        "sequence": result_dict["aa_seq"],
        "label": result_dict["label"],
        "pred_label": result_dict["pred_label"],
        "probability": result_dict["probability"]
    })
    df.to_csv(args.output_csv, index=False)
    print(f"Saved standardized ProtSolM results to {args.output_csv}")

if __name__ == "__main__":
    main()
