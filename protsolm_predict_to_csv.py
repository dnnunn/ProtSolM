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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load features
    logger.info("***** Loading Feature *****")
    feature_df = pd.read_csv(args.feature_file)
    feature_dict = {}
    for i, row in feature_df.iterrows():
        # Use the correct key and drop the correct column based on your CSV header
        key = row["protein name"]
        feature_dict[key] = row.drop("protein name").values.astype(float)
    args.feature_dim = len(feature_df.columns) - 1

    # Load test set
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
