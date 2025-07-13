#!/usr/bin/env python3
"""
ProtSolM Predictor Batch Wrapper
Standardizes output for benchmarking solubility predictors.

Usage:
  python protsolm_predict_wrapper.py --fasta <input_fasta> --out <output_csv>

Outputs CSV with columns:
  Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble
"""
import argparse
import os
import subprocess
import pandas as pd
from Bio import SeqIO

def fasta_to_csv(fasta_path, csv_path):
    # ProtSolM expects: name,aa_seq
    records = list(SeqIO.parse(fasta_path, "fasta"))
    df = pd.DataFrame({
        'name': [rec.id for rec in records],
        'aa_seq': [str(rec.seq) for rec in records]
    })
    df.to_csv(csv_path, index=False)
    return df

def run_protsolm(input_csv, output_dir):
    # Get the ProtSolM root directory
    protsolm_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = 'eval.py'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use absolute paths for input and output
    abs_input_csv = os.path.abspath(input_csv)
    abs_output_dir = os.path.abspath(output_dir)
    
    # Save current directory
    cwd = os.getcwd()
    
    try:
        # Change to ProtSolM directory before running eval.py
        os.chdir(protsolm_dir)
        
        # Find a valid dataset directory with pdb or esmfold_pdb subdirectory
        dataset_paths = [
            os.path.join(protsolm_dir, 'data', 'PDBSol'),
            os.path.join(protsolm_dir, 'data', 'ExternalTest')
        ]
        
        valid_dataset = None
        for path in dataset_paths:
            if os.path.exists(os.path.join(path, 'pdb')) or os.path.exists(os.path.join(path, 'esmfold_pdb')):
                valid_dataset = path
                dataset_name = os.path.basename(path)
                break
        
        if valid_dataset is None:
            raise ValueError("No valid dataset directory found with pdb or esmfold_pdb subdirectory. " 
                           "Please make sure one exists in the PDBSol or ExternalTest directories.")
        
        print(f"Using dataset: {valid_dataset}")
        feature_file = os.path.join(valid_dataset, f"{dataset_name}_feature.csv")
        
        if not os.path.exists(feature_file):
            # If feature file doesn't exist, look for any feature file
            print(f"Feature file {feature_file} not found, looking for any feature file")
            feature_files = [f for f in os.listdir(valid_dataset) if f.endswith('_feature.csv')]
            if feature_files:
                feature_file = os.path.join(valid_dataset, feature_files[0])
                print(f"Found feature file: {feature_file}")
            else:
                raise ValueError(f"No feature file found in {valid_dataset}")
        
        # Run the command with all required parameters per README
        cmd = [
            'python', script_name,
            '--supv_dataset', valid_dataset,
            '--test_file', abs_input_csv,
            '--test_result_dir', abs_output_dir,
            '--feature_file', feature_file,
            '--feature_name', 'aa_composition', 'gravy', 'ss_composition', 'hygrogen_bonds', 'exposed_res_fraction', 'pLDDT',
            '--use_plddt_penalty',
            '--batch_token_num', '3000'
        ]
        
        print(f"Running ProtSolM from {os.getcwd()}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    finally:
        # Restore original directory
        os.chdir(cwd)
    
    return os.path.join(abs_output_dir, 'test_result.csv')

def main():
    parser = argparse.ArgumentParser(description="Batch ProtSolM predictor wrapper")
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    parser.add_argument('--out', '-o', required=True, help='Output CSV file')
    args = parser.parse_args()

    tmp_input_csv = 'tmp_protsolm_input.csv'
    tmp_output_dir = 'tmp_protsolm_results'

    input_df = fasta_to_csv(args.fasta, tmp_input_csv)
    result_csv = run_protsolm(tmp_input_csv, tmp_output_dir)

    df = pd.read_csv(result_csv)
    df['Predictor'] = 'ProtSolM'
    # If probability is not available, set SolubilityScore as 1 for soluble, 0 for insoluble
    if 'predicted_probability' in df.columns:
        df['SolubilityScore'] = df['predicted_probability'].astype(float)
        df['Probability_Soluble'] = df['SolubilityScore']
        df['Probability_Insoluble'] = 1 - df['SolubilityScore']
    else:
        df['SolubilityScore'] = df['pred_label'].map(lambda x: 1 if x == 1 or str(x).lower() == 'soluble' else 0)
        df['Probability_Soluble'] = df['SolubilityScore']
        df['Probability_Insoluble'] = 1 - df['SolubilityScore']
    df.rename(columns={'name': 'Accession', 'aa_seq': 'Sequence'}, inplace=True)
    df = df[['Accession', 'Sequence', 'Predictor', 'SolubilityScore', 'Probability_Soluble', 'Probability_Insoluble']]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Results written to {args.out}")

if __name__ == '__main__':
    main()
