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
    script_path = os.path.join(os.path.dirname(__file__), 'eval.py')
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'python', script_path,
        '--test_file', input_csv,
        '--test_result_dir', output_dir
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, 'test_result.csv')

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
