# ProtSolM: Paper-Derived Implementation & Integration Notes

_Last updated: 2025-07-16_

---

## 1. **Model Overview and Architecture**
- **ProtSolM** is a multi-modal deep learning model for protein solubility prediction.
- Integrates three information sources:
  1. **Sequence** (AA-level, ESM2 embeddings)
  2. **Structure** (kNN graph of protein backbone, processed by EGNN)
  3. **Physicochemical features** (42 handcrafted, DSSP/mkdssp-derived)
- Model pipeline:
  - **Pre-training:** ESM2 + EGNN layers on wild-type proteins (denoising task)
  - **Fine-tuning:** Attention pooling + concatenation with normalized protein-level features; fully connected layers for binary solubility prediction

---

## 2. **Feature Extraction Pipeline**
- **All features must be generated and processed identically for both training and inference.**
- **Protein-level features** (total = 42):
  - Fraction of charged AAs: C, D, E, R, H
  - Fraction of turn-forming residues: N, G, P
  - GRAVY index (hydropathy)
  - Secondary structure fractions (DSSP 3-state, 9-state)
  - Exposed residue fractions (RSA bins from 5% to 100%)
  - Hydrogen bond count/density (MDTraj)
  - Structure confidence (mean pLDDT, from ESMFold)
- **Sequence features:** ESM2 embeddings (handled inside model)
- **Structure features:** kNN graph, edge features (distances, local positions, sequence encoding)
- **Critical:**
  - Use **mkdssp** for secondary structure/solvent accessibility
  - Use the same `get_feature.py` script and mkdssp version as in training
  - Columns and order in feature CSV must match training exactly

---

## 3. **Data and Preprocessing**
- **Training data:** PDBSol (PDBS OL) — >60,000 proteins, processed as follows:
  - Remove non-protein entities (virus, DNA/RNA, etc.)
  - Remove His-tags, redundancy (MMS EQS 2, 25% identity), extreme lengths
  - Exclude transmembrane proteins (DeepTMHMM)
  - Balance by length/class
- **Test data:** External benchmarks (ESOL-agg, NESG-SoluProt, NESG-DSResSol), deduplicated at 25% identity
- **All structures predicted with ESMFold**
- **All features extracted with mkdssp, MDTraj, biotite, etc.**

---

## 4. **Integration with PeptideFrontEnd & GA**
- **Workflow:**
  1. Generate/collect PDBs for all candidate sequences
  2. Run mkdssp to extract DSSP features
  3. Run `get_feature.py` to produce feature CSV (42 features, canonical order)
  4. Use ProtSolM wrapper to predict solubility (outputs standardized CSV)
  5. PeptideFrontEnd/GA calls ProtSolM as a subprocess, parses CSV, uses score in fitness evaluation
- **Fitness function:**
  - Use ProtSolM solubility score (alongside bioactive ratio, cleavage success, etc.)
  - Ensure feature file for each batch matches training format
- **Automation:**
  - Automate all steps above for batch processing in GA
  - Standardize input/output formats for benchmarking and integration

---

## 5. **Model Training & Evaluation Protocols**
- **Pre-training:** Denoising task with ESM2 + EGNN
- **Fine-tuning:**
  - AdamW optimizer, lr=0.0005, weight decay=0.01, dropout=0.1
  - Dynamic batching (up to 16,000 tokens per batch)
  - 30 epochs max, patience=5 (early stopping on validation ACC)
- **Evaluation:**
  - Metrics: ACC, precision, recall, AUC, MCC
  - Binary threshold: 0.5
  - Compare to baselines: DeepSoluE, ccSOL omics, SoluProt, SKADE, Camsol, NetSolP, DSResSOL, ESM2, ProtBert, ProtT5, Ankh

---

## 6. **Ablation Study Insights**
- Removing any of: pLDDT penalty, attention pooling, or physicochemical features reduces accuracy
- **All three modalities (sequence, structure, physicochemical) are critical for SOTA performance**

---

## 7. **Best Practices & Recommendations**
- **Never change the feature extraction pipeline** between training and inference
- **Document and enforce canonical feature order** in all scripts
- **Automate and standardize the pipeline** for GA/PeptideFrontEnd integration
- **Keep mkdssp and all dependencies version-matched** to training
- **Keep all wrappers and output CSVs standardized** for benchmarking

---

## 8. **References**
- Paper: Tan et al., 2024, "ProtSolM: Protein Solubility Prediction with Multi-modal Features" (arXiv:2406.19744)
- Code: https://github.com/tyang816/ProtSolM
- ESM2: Lin et al., Science 2023
- EGNN: Satorras et al., ICML 2021

---

**This document summarizes all implementation-critical information from the Tan2024_ProtSolM paper for current and future development, integration, and troubleshooting. Update as needed when the pipeline or requirements change.**
