import argparse
import warnings
import torch
import os
import sys
import yaml
import datetime
import logging
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import *
# from transformers import get_inverse_sqrt_schedule
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from accelerate.utils import set_seed
from accelerate import Accelerator
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef
from src.models import ProtssnClassification, PLM_model, GNN_model
from src.utils.data_utils import BatchSampler
from src.utils.utils import param_num, total_param_num
from src.dataset.supervise_dataset import SuperviseDataset
from src.utils.dataset_utils import NormalizeProtein

# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
# ignore warning information
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 3 + "%s" % nowtime + "==========" * 3)
    print(str(info) + "\n")

class StepRunner:
    def __init__(self, args, model, 
                 loss_fn, accelerator=None,
                 metrics_dict=None,
                 ):
        self.model = model
        self.metrics_dict = metrics_dict
        self.accelerator = accelerator
        self.loss_fn = loss_fn
        self.args = args

    def step(self, batch):        
        logits, ssn_emebds = self.model(plm_model, gnn_model, batch, True)
        logits = logits.cuda()
        label = torch.cat([data.label for data in batch]).to(logits.device)
        pred_labels = torch.argmax(logits, 1).cpu().numpy()
        loss = self.loss_fn(logits, label)
        # compute metrics
        for name, metric_fn in self.metrics_dict.items():
            metric_fn.update(torch.argmax(logits, 1), label)
        return loss.item(), self.model, self.metrics_dict, pred_labels, ssn_emebds

    def train_step(self, batch):
        self.model.train()
        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        return self.step(batch)

    def __call__(self, batch):
        return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.args = steprunner.args

    def __call__(self, dataloader):
        loop = tqdm(dataloader, total=len(dataloader), file=sys.stdout)
        total_loss = 0
        result_dict = {'name':[], 'aa_seq':[], 'label':[], 'pred_label':[]}
        ssn_embeds = []
        for batch in loop:
            step_loss, model, metrics_dict, pred_label, ssn_embed = self.steprunner(batch)
            # Debug: Print raw logits for the batch and compute probabilities
            with torch.no_grad():
                logits, _ = model(plm_model, gnn_model, batch, True)
                print("[DEBUG] Raw logits for batch:", logits.detach().cpu().numpy())
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # Probability of soluble (class 1)
                print("[DEBUG] Soluble probabilities for batch:", probs)
            result_dict["pred_label"].extend(pred_label)
            if "probability" not in result_dict:
                result_dict["probability"] = []
            result_dict["probability"].extend(probs.tolist())
            result_dict["name"].extend([data.name for data in batch])
            result_dict["aa_seq"].extend([data.aa_seq for data in batch])
            result_dict["label"].extend([data.label.item() for data in batch])
            ssn_embeds.append(ssn_embed)
            step_log = dict({f"eval/loss": round(step_loss, 3)})
            loop.set_postfix(**step_log)
            total_loss += step_loss
        ssn_embeds = torch.cat(ssn_embeds, dim=0)
        epoch_metric_results = {}
        for name, metric_fn in metrics_dict.items():
            epoch_metric_results[f"eval/{name}"] = metric_fn.compute().item()
            metric_fn.reset()
        avg_loss = total_loss / len(dataloader)
        epoch_metric_results[f"eval/loss"] = avg_loss
        return model, epoch_metric_results, result_dict, ssn_embeds

def eval_model(args, model, loss_fn, 
                accelerator=None, metrics_dict=None, 
                test_data=None
                ):
    model_path = os.path.join(args.model_dir, args.model_name)        
    if test_data:
        test_step_runner = StepRunner(
            args=args, model=model, 
            loss_fn=loss_fn, accelerator=accelerator,
            metrics_dict=deepcopy(metrics_dict), 
        )
        epoch_runner = EpochRunner(test_step_runner)
        model, epoch_metric_results, result_dict, ssn_embeds = epoch_runner(test_data)
        for name, metric in epoch_metric_results.items():
            epoch_metric_results[name] = [metric]
            print(f">>> {name}: {'%.3f'%metric}")
    
    if args.test_result_dir:
        os.makedirs(args.test_result_dir, exist_ok=True)
        pd.DataFrame(result_dict).to_csv(f"{args.test_result_dir}/test_result.csv", index=False)
        pd.DataFrame(epoch_metric_results).to_csv(f"{args.test_result_dir}/test_metrics.csv", index=False)
        torch.save(ssn_embeds, f"{args.test_result_dir}/ssn_embeds.pt")

def create_parser():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_hidden_dim", type=int, default=512, help="hidden size of gnn")
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    parser.add_argument("--plm_hidden_size", type=int, default=1280, help="hidden size of plm")
    parser.add_argument("--pooling_method", type=str, default="attention1d", help="pooling method")
    parser.add_argument("--pooling_dropout", type=float, default=0.1, help="pooling dropout")
    
    # training strategy
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay")
    parser.add_argument("--batch_token_num", type=int, default=4096, help="how many tokens in one batch")
    parser.add_argument("--max_graph_token_num", type=int, default=3000, help="max token num a graph has")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="clip grad norm")
    
    # dataset
    parser.add_argument("--num_labels", type=int, default=2, help="number of labels")
    parser.add_argument("--problem_type", type=str, default="classification", help="classification or regression")
    parser.add_argument("--supv_dataset", type=str, help="supervise protein dataset")
    parser.add_argument("--test_file", type=str, help="test label file")
    parser.add_argument('--test_result_dir', type=str, default=None, help='test result directory')
    parser.add_argument("--feature_file", type=str, default=None, help="feature file")
    parser.add_argument("--feature_name", nargs="+", default=None, help="feature names")
    parser.add_argument("--feature_dim", type=int, default=0, help="feature dim")
    parser.add_argument("--feature_embed_dim", type=int, default=512, help="feature embed dim")
    parser.add_argument("--use_plddt_penalty", action="store_true", help="use plddt penalty")
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=20, help="graph dataset K")
    parser.add_argument("--gnn_model_path", type=str, default="./model/protssn_k20_h512.pt", help="gnn model path")
    
    # load model
    parser.add_argument("--model_dir", type=str, default="./ckpt", help="model save dir")
    parser.add_argument("--model_name", type=str, default="feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt", help="model name")

    args = parser.parse_args()
    return args

feature_dict ={}

if __name__ == "__main__":
    args = create_parser()

    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    args.gnn_config["hidden_channels"] = args.gnn_hidden_dim

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.feature_file:
        logger.info("***** Loading Feature *****")
        # === DEFINITIVE FIX START ===
        # Load features and set the protein name as the index for easy lookup.
        feature_df = pd.read_csv(args.feature_file).set_index('protein name')

        # Drop the 'L' column as it is not a feature for the model.
        if 'L' in feature_df.columns:
            feature_df = feature_df.drop('L', axis=1)

        # The remaining columns are our features.
        feature_columns = feature_df.columns
        args.feature_dim = len(feature_columns)
        logger.info(f"Correctly calculated feature dimension: {args.feature_dim}")

        # Create the feature dictionary for the model.
        for protein_name, row in tqdm(feature_df.iterrows()):
            feature_dict[protein_name] = row.values.astype(np.float32)
        # === DEFINITIVE FIX END ===

        if type(args.feature_name) != list:
            args.feature_name = [args.feature_name]

        feature_aa_composition = ["1-C", "1-D", "1-E", "1-R", "1-H", "Turn-forming residues fraction"]
        if "aa_composition" in args.feature_name:
            aa_composition_df = feature_df[feature_aa_composition]
            args.feature_dim += len(feature_aa_composition)
        
        feature_gravy = ["GRAVY"]
        if "gravy" in args.feature_name:
            gravy_df = feature_df[feature_gravy]
            args.feature_dim += len(feature_gravy)
        
        feature_ss_composition = ["ss8-G", "ss8-H", "ss8-I", "ss8-B", "ss8-E", "ss8-T", "ss8-S", "ss8-P", "ss8-L", "ss3-H", "ss3-E", "ss3-C"]
        if "ss_composition" in args.feature_name:
            ss_composition_df = feature_df[feature_ss_composition]
            args.feature_dim += len(feature_ss_composition)
        
        feature_hygrogen_bonds = ["Hydrogen bonds", "Hydrogen bonds per 100 residues"]
        if "hygrogen_bonds" in args.feature_name:
            hygrogen_bonds_df = feature_df[feature_hygrogen_bonds]
            args.feature_dim += len(feature_hygrogen_bonds)
        
        feature_exposed_res_fraction = [
            "Exposed residues fraction by 5%", "Exposed residues fraction by 10%", "Exposed residues fraction by 15%", 
            "Exposed residues fraction by 20%", "Exposed residues fraction by 25%", "Exposed residues fraction by 30%", 
            "Exposed residues fraction by 35%", "Exposed residues fraction by 40%", "Exposed residues fraction by 45%", 
            "Exposed residues fraction by 50%", "Exposed residues fraction by 55%", "Exposed residues fraction by 60%", 
            "Exposed residues fraction by 65%", "Exposed residues fraction by 70%", "Exposed residues fraction by 75%", 
            "Exposed residues fraction by 80%", "Exposed residues fraction by 85%", "Exposed residues fraction by 90%", 
            "Exposed residues fraction by 95%", "Exposed residues fraction by 100%"
            ]
        if "exposed_res_fraction" in args.feature_name:
            exposed_res_fraction_df = feature_df[feature_exposed_res_fraction]
            args.feature_dim += len(feature_exposed_res_fraction)
        
        feature_pLDDT = ["pLDDT"]
        if "pLDDT" in args.feature_name:
            plddt_df = feature_df[feature_pLDDT]
            args.feature_dim += len(feature_pLDDT)
        
        
        for i in tqdm(range(len(feature_df))):
            name = feature_df.index[i].split(".")[0]
            feature_dict[name] = []
            if "aa_composition" in args.feature_name:
                feature_dict[name] += list(aa_composition_df.iloc[i])
            if "gravy" in args.feature_name:
                feature_dict[name] += list(gravy_df.iloc[i])
            if "ss_composition" in args.feature_name:
                feature_dict[name] += list(ss_composition_df.iloc[i])
            if "hygrogen_bonds" in args.feature_name:
                feature_dict[name] += list(hygrogen_bonds_df.iloc[i])
            if "exposed_res_fraction" in args.feature_name:
                feature_dict[name] += list(exposed_res_fraction_df.iloc[i])
            if "pLDDT" in args.feature_name:
                feature_dict[name] += list(plddt_df.iloc[i])
    
    # load dataset
    logger.info("***** Loading Dataset *****")
    datatset_name = args.supv_dataset.split("/")[-1]
    if os.path.exists(f"{args.supv_dataset}/esmfold_pdb"):
        pdb_dir = f"{args.supv_dataset}/esmfold_pdb"
    elif os.path.exists(f"{args.supv_dataset}/pdb"):
        pdb_dir = f"{args.supv_dataset}/pdb"
    else:
        raise ValueError("No pdb or esmfold_pdb directory found in the dataset")
    graph_dir = f"{datatset_name}_k{args.c_alpha_max_neighbors}"
    supervise_dataset = SuperviseDataset(
        root=args.supv_dataset,
        raw_dir=pdb_dir,
        name=graph_dir,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pre_transform=NormalizeProtein(
            filename=f'norm/cath_k{args.c_alpha_max_neighbors}_mean_attr.pt'
        ),
    )

    label_dict, seq_dict = {}, {}
    def get_dataset(df):
        names, node_nums = [], []
        for name, label, seq in zip(df["id"], df["label"], df["aa_seq"]):
            names.append(name)
            label_dict[name] = label
            seq_dict[name] = seq
            node_nums.append(len(seq))
        return names, node_nums
    test_df = pd.read_csv(args.test_file)
    test_names, test_node_nums = get_dataset(test_df)
    # Debug: Print unique labels and their counts
    print("[DEBUG] Test label distribution:", test_df['label'].value_counts().to_dict())
    
    # multi-thread load data will shuffle the order of data
    # so we need to save the information
    def process_data(name, fd):
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
    
    def collect_fn(batch):
        batch_data = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            feature_d =deepcopy(feature_dict)
            futures = [executor.submit(process_data, name, feature_d) for name in batch]
            for future in as_completed(futures):
                graph = future.result()
                batch_data.append(graph)
        return batch_data
    
    test_dataloader = DataLoader(
        dataset=test_names, num_workers=4, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            node_num=test_node_nums,
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=False
            )
        )
    
    logger.info("***** Load Model *****")
    # load model
    global plm_model
    plm_model = PLM_model(args).to(device)
    global gnn_model
    gnn_model = GNN_model(args).to(device)
    
    # Load checkpoint and extract state_dict if it's in a dictionary format
    checkpoint = torch.load(args.gnn_model_path)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove BatchNorm keys if present (for compatibility with LayerNorm model)
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
    loss_fn = torch.nn.CrossEntropyLoss()
    # Load state dict with strict=False and print missing/unexpected keys
    missing, unexpected = protssn_classification.load_state_dict(filtered_state_dict, strict=False)
    print("[DEBUG] Missing keys:", missing)
    print("[DEBUG] Unexpected keys:", unexpected)
    
    for param in plm_model.parameters():
        param.requires_grad = False
    for param in gnn_model.parameters():
        param.requires_grad = False
    logger.info(total_param_num(protssn_classification))
    logger.info(param_num(protssn_classification))

    accelerator = Accelerator()
    protssn_classification, test_dataloader = accelerator.prepare(
        protssn_classification, test_dataloader
    )
    metrics_dict = {
        "acc": BinaryAccuracy().to(device),
        "recall": BinaryRecall().to(device),
        "precision": BinaryPrecision().to(device),
        "mcc": BinaryMatthewsCorrCoef().to(device),
        "auroc": BinaryAUROC().to(device),
        "f1": BinaryF1Score().to(device),
    }
    
    logger.info("***** Running eval *****")
    logger.info("  Num test examples = %d", len(test_names))
    logger.info("  Batch token num = %d", args.batch_token_num)
    
    eval_model(
        args=args, model=protssn_classification, 
        loss_fn=loss_fn, 
        accelerator=accelerator, metrics_dict=metrics_dict, 
        test_data=test_dataloader
        )
    