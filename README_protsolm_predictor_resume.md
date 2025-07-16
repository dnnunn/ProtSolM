# ProtSolM Predictor Integration & Debugging: Resumption Checklist

_Last updated: 2025-07-15_

## Objective
Fully integrate and run the ProtSolM protein solubility prediction pipeline for benchmarking, ensuring feature file and model checkpoint compatibility, and standardized CSV output.

---

## Current Status
- **Prediction script (`protsolm_predict_to_csv.py`)**: Patched to robustly load features and check for dimension mismatches.
- **Feature file in use**: `data/ExternalTest/ExternalTest_feature.csv` (header pasted below)
- **Model checkpoint**: `ckpt/feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt`
- **Observed error**: Feature dimension mismatch (script sees 1322, checkpoint expects 1792).
- **Script now halts with a clear error if dimensions do not match.**

---

## What Must Be Done to Resume

### 1. **Obtain the Correct Training Feature File**
- The model checkpoint expects a feature vector of length **1792**.
- The current feature file only provides ~41 features (excluding 'L' and 'protein name').
- The only way to resolve the mismatch is to obtain the **exact feature file used during training** (or its header).
- **Action:**
  - Locate the canonical training feature file (likely on the VM or in backup).
  - If found, paste or save its header for comparison.
  - If not, reconstruct the full feature list from the training code.

### 2. **Patch or Reformat Your Feature File**
- Once you have the correct header, reformat your current feature file to:
    - Include all required columns (in the correct order)
    - Fill missing columns with zeros/NaN if necessary
    - Remove extra/unexpected columns
- I can provide a Python script to automate this reformatting once you have the header.

### 3. **Run the Prediction Script**
- With the correct feature file in place, rerun:
  ```bash
  python protsolm_predict_to_csv.py \
    --supv_dataset data/PDBSol \
    --test_file data/PDBSol/test_sequences.csv \
    --feature_file data/ExternalTest/ExternalTest_feature.csv \
    --gnn_model_path ckpt/feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt \
    --gnn egnn \
    --gnn_hidden_dim 512 \
    --plm facebook/esm2_t33_650M_UR50D \
    --output_csv results/protsolm_pred.csv
  ```
- The script will now halt with a clear error if the feature dimension is wrong.

### 4. **If Still Stuck**
- Check `eval.py` for how features are loaded and what columns are expected.
- Confirm that the order and count of columns in the feature file matches exactly what is used in training.
- If you need help, paste the canonical header here and request a reformatting script.

---

## Reference: Current Feature File Header
```
L,1-C,1-D,1-E,1-R,1-H,Turn-forming residues fraction,GRAVY,ss8-G,ss8-H,ss8-I,ss8-B,ss8-E,ss8-T,ss8-S,ss8-P,ss8-L,ss3-H,ss3-E,ss3-C,Hydrogen bonds,Hydrogen bonds per 100 residues,Exposed residues fraction by 5%,Exposed residues fraction by 10%,Exposed residues fraction by 15%,Exposed residues fraction by 20%,Exposed residues fraction by 25%,Exposed residues fraction by 30%,Exposed residues fraction by 35%,Exposed residues fraction by 40%,Exposed residues fraction by 45%,Exposed residues fraction by 50%,Exposed residues fraction by 55%,Exposed residues fraction by 60%,Exposed residues fraction by 65%,Exposed residues fraction by 70%,Exposed residues fraction by 75%,Exposed residues fraction by 80%,Exposed residues fraction by 85%,Exposed residues fraction by 90%,Exposed residues fraction by 95%,Exposed residues fraction by 100%,pLDDT,protein name
```

---

## Additional Notes
- The `protsolm_predict_to_csv.py` script now includes a robust feature dimension check and will refuse to run if the feature vector does not match the checkpoint.
- The pipeline requires the feature file, test set, and checkpoint to be in perfect agreement.
- If you want to automate CSV reformatting, save the canonical header and request a script to do so.
- The benchmarking pipeline expects standardized CSV output: `id`, `sequence`, `label`, `pred_label`, `probability`.

---

## Key Paths
- **Feature file:** `data/ExternalTest/ExternalTest_feature.csv`
- **Test set:** `data/PDBSol/test_sequences.csv`
- **Model checkpoint:** `ckpt/feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt`
- **Output CSV:** `results/protsolm_pred.csv`
- **GNN config YAML:** `src/config/egnn.yaml`

---

## When You Return
1. Obtain or reconstruct the canonical feature file/header.
2. Patch/reformat your feature file to match.
3. Rerun the prediction script.
4. If stuck, request a reformatting script or further debugging help.

---

**This file is your quickstart guide to resuming the ProtSolM integration and debugging process.**
