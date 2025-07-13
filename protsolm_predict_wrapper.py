#!/usr/bin/env python3
"""
ProtSolM Predictor Batch Wrapper
Standardizes output for benchmarking solubility predictors.

Usage:
  python protsolm_predict_wrapper.py --fasta <input_fasta> --out <output_csv>

Outputs CSV with columns:
  Accession, Sequence, Predictor, SolubilityScore, Probability_Soluble, Probability_Insoluble
"""
import os
import sys
import argparse
import subprocess
import shutil
import tempfile
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

def setup_custom_dataset(input_csv, structures_dir=None):
    """Set up a custom dataset directory with PDB files for ProtSolM to use"""
    # Get the ProtSolM root directory
    protsolm_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define custom dataset location
    custom_dataset_dir = os.path.join(protsolm_dir, 'custom_dataset')
    pdb_dir = os.path.join(custom_dataset_dir, 'pdb')
    
    # Create directories if they don't exist
    os.makedirs(custom_dataset_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)
    
    print(f"Set up custom dataset directory at: {custom_dataset_dir}")
    
    # Copy the input CSV as test.csv in the custom dataset directory
    shutil.copy(input_csv, os.path.join(custom_dataset_dir, 'test.csv'))
    
    # If structures_dir is provided, symlink PDB files to the custom dataset's pdb directory
    if structures_dir and os.path.exists(structures_dir):
        print(f"Linking PDB files from {structures_dir} to {pdb_dir}")
        try:
            # Get list of PDB files in structures_dir
            pdb_files = [f for f in os.listdir(structures_dir) if f.endswith('.pdb')]
            print(f"Found {len(pdb_files)} PDB files in {structures_dir}")
            
            # Link each PDB file to the custom pdb directory
            for pdb_file in pdb_files:
                src = os.path.join(structures_dir, pdb_file)
                dst = os.path.join(pdb_dir, pdb_file)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
            print(f"Linked {len(pdb_files)} PDB files to custom dataset directory")
        except Exception as e:
            print(f"Error linking PDB files: {e}")
    
    # If a feature file exists in one of the built-in datasets, copy it to the custom dataset
    # This is a workaround until we generate features for our custom PDB files
    for dataset_name in ['PDBSol', 'ExternalTest']:
        src_path = os.path.join(protsolm_dir, 'data', dataset_name)
        if os.path.exists(src_path):
            feature_files = [f for f in os.listdir(src_path) if f.endswith('_feature.csv')]
            if feature_files:
                src_feature = os.path.join(src_path, feature_files[0])
                dst_feature = os.path.join(custom_dataset_dir, 'custom_feature.csv')
                shutil.copy(src_feature, dst_feature)
                print(f"Copied feature file from {src_feature} to {dst_feature}")
                break
    
    return custom_dataset_dir

def run_protsolm(input_csv, output_dir, structures_dir=None):
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
        
        # Setup custom dataset with PDB files
        if structures_dir:
            structures_abs_path = os.path.abspath(structures_dir)
        else:
            # Try to find structures in standard location relative to input CSV
            input_dir = os.path.dirname(abs_input_csv)
            structures_abs_path = os.path.join(input_dir, 'test_results', 'structures')
            if not os.path.exists(structures_abs_path):
                structures_abs_path = None
                print("No structures directory found, will use existing dataset")
            
        custom_dataset_dir = setup_custom_dataset(abs_input_csv, structures_abs_path)
        
        # Use the custom dataset feature file
        feature_file = os.path.join(custom_dataset_dir, 'custom_feature.csv')
        
        # Run the command with all required parameters per README
        cmd = [
            'python', script_name,
            '--supv_dataset', custom_dataset_dir,
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ProtSolM solubility prediction wrapper')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file path')
    parser.add_argument('--out', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--structures', type=str, help='Directory containing PDB structure files')
    args = parser.parse_args()
    
    # Ensure both input and output paths exist or can be created
    if not os.path.exists(args.fasta):
        print(f"Error: Input FASTA file not found: {args.fasta}")
        return 1
        
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if structures directory exists if provided
    structures_dir = None
    if args.structures:
        if not os.path.exists(args.structures):
            print(f"Warning: Structures directory not found: {args.structures}")
        else:
            structures_dir = args.structures
            print(f"Using structures from: {structures_dir}")
    
    # Convert FASTA to CSV if needed
    print(f"Processing FASTA: {args.fasta}")
    temp_dir = tempfile.mkdtemp()
    input_csv = os.path.join(temp_dir, 'input.csv')
    fasta_to_csv(args.fasta, input_csv)
    
    try:
        # Run ProtSolM prediction
        result_csv = run_protsolm(input_csv, temp_dir, structures_dir)
        
        # Copy result to output path
        if os.path.exists(result_csv):
            shutil.copyfile(result_csv, args.out)
            print(f"Prediction results saved to: {args.out}")
        else:
            print(f"Error: ProtSolM did not generate results at {result_csv}")
            return 1
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0

if __name__ == '__main__':
    main()
