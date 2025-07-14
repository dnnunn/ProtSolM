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
import logging
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fasta_to_csv(fasta_path, csv_path):
    """Convert a FASTA file to a CSV file with columns name, aa_seq, and label"""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    
    # Create dataframe
    data = []
    for record in records:
        data.append({
            "name": record.id,
            "aa_seq": str(record.seq),
            # Add a default label column (1 = soluble, 0 = insoluble)
            # Since we're making predictions, we'll use a dummy value
            # that will be replaced by the actual prediction
            "label": 1  # Default to soluble as a placeholder
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return df

def create_fallback_feature_file(feature_file_path, input_csv_path):
    """Create a fallback feature file with default values if feature generation failed"""
    import pandas as pd
    
    logging.warning(f"Creating FALLBACK feature file with ZERO VALUES: {feature_file_path}")
    logging.warning(f"NOTE: These are placeholder features and may affect prediction accuracy")
    
    print(f"Creating fallback feature file at {feature_file_path}")
    
    # Read the input CSV to get protein names
    input_df = pd.read_csv(input_csv_path)
    
    # Try different column names for protein identifiers
    id_column = None
    for col in ['seq_id', 'id', 'name', 'protein', 'protein_id', 'sequence_id']:
        if col in input_df.columns:
            id_column = col
            break
    
    # If no standard column is found, just use the first column
    if id_column is None:
        print(f"Warning: No standard ID column found in {input_csv_path}. Using first column.")
        id_column = input_df.columns[0]
        
    print(f"Using '{id_column}' column for protein identifiers")
    protein_names = input_df[id_column].tolist()
    
    # Create a dictionary with default values
    feature_data = {'protein name': protein_names}
    
    # Exactly match the feature names from eval.py
    
    # Amino acid composition features (from eval.py line ~193)
    aa_composition_features = [
        '1-C', '1-D', '1-E', '1-R', '1-H', 'Turn-forming residues fraction'
    ]
    
    # GRAVY features (from eval.py line ~198) - Note: uppercase
    gravy_features = ['GRAVY']
    
    # Secondary structure composition features (from eval.py line ~203)
    ss_composition_features = [
        'ss8-G', 'ss8-H', 'ss8-I', 'ss8-B', 'ss8-E', 'ss8-T', 'ss8-S', 'ss8-P', 'ss8-L',
        'ss3-H', 'ss3-E', 'ss3-C'
    ]
    
    # Hydrogen bond features (from eval.py line ~208)
    hydrogen_bond_features = [
        'Hydrogen bonds', 'Hydrogen bonds per 100 residues'
    ]
    
    # Exposed residue fraction features (from eval.py line ~213)
    exposed_res_features = [
        'Exposed residues fraction by 5%', 'Exposed residues fraction by 10%', 'Exposed residues fraction by 15%',
        'Exposed residues fraction by 20%', 'Exposed residues fraction by 25%', 'Exposed residues fraction by 30%',
        'Exposed residues fraction by 35%', 'Exposed residues fraction by 40%', 'Exposed residues fraction by 45%',
        'Exposed residues fraction by 50%', 'Exposed residues fraction by 55%', 'Exposed residues fraction by 60%',
        'Exposed residues fraction by 65%', 'Exposed residues fraction by 70%', 'Exposed residues fraction by 75%',
        'Exposed residues fraction by 80%', 'Exposed residues fraction by 85%', 'Exposed residues fraction by 90%',
        'Exposed residues fraction by 95%', 'Exposed residues fraction by 100%'
    ]
    
    # pLDDT features (from eval.py line ~226) - Note: mixed case
    plddt_features = ['pLDDT']
    
    # Additional features for potential compatibility
    additional_features = []
    
    # Combine all features
    all_features = aa_composition_features + ss_composition_features + \
                   hydrogen_bond_features + exposed_res_features + \
                   plddt_features + gravy_features + additional_features
    
    # Add default feature values (zeros) for each feature
    for feature in all_features:
        feature_data[feature] = [0.0] * len(protein_names)
    
    # Create the DataFrame and save to CSV
    feature_df = pd.DataFrame(feature_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(feature_file_path), exist_ok=True)
    
    # Save to CSV
    feature_df.to_csv(feature_file_path, index=False)
    
    print(f"Fallback feature file created with {len(all_features)} features for {len(protein_names)} proteins")
    print(f"Fallback feature file saved to {feature_file_path}")
    
    return feature_file_path

def setup_custom_dataset(input_csv, structures_dir=None):
    """Set up a custom dataset directory with PDB files for ProtSolM to use"""
    # Get the ProtSolM root directory
    protsolm_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define custom dataset location
    custom_dataset_dir = os.path.join(protsolm_dir, 'custom_dataset')
    os.makedirs(custom_dataset_dir, exist_ok=True)
    
    # Create PDB directory
    pdb_dir = os.path.join(custom_dataset_dir, 'pdb')
    os.makedirs(pdb_dir, exist_ok=True)
    
    # Copy the input CSV as test.csv in the custom dataset directory
    shutil.copy(input_csv, os.path.join(custom_dataset_dir, 'test.csv'))
    
    # Link PDB files from structures_dir to custom_dataset_dir/pdb if provided
    if structures_dir and os.path.exists(structures_dir):
        print(f"Linking PDB files from {structures_dir} to {pdb_dir}")
        try:
            # Get list of PDB files in structures_dir
            pdb_files = [f for f in os.listdir(structures_dir) if f.endswith('.pdb')]
            print(f"Found {len(pdb_files)} PDB files in {structures_dir}")
            
            # Link each PDB file to the custom pdb directory
            linked_count = 0
            for pdb_file in pdb_files:
                src = os.path.join(structures_dir, pdb_file)
                dst = os.path.join(pdb_dir, pdb_file)
                
                # Check if source is a file or directory
                if os.path.isfile(src):
                    # It's a regular file, just link it
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
                        linked_count += 1
                elif os.path.isdir(src):
                    # It's a directory - find a PDB file inside it
                    print(f"Warning: {pdb_file} is a directory, not a file")
                    inner_pdb_files = [f for f in os.listdir(src) if f.endswith('.pdb')]
                    if inner_pdb_files:
                        # Use the first PDB file found
                        inner_src = os.path.join(src, inner_pdb_files[0])
                        print(f"Using {inner_pdb_files[0]} from directory {pdb_file}")
                        shutil.copy(inner_src, dst)
                        linked_count += 1
                    else:
                        print(f"Warning: No PDB files found in directory {src}")
            
            print(f"Processed {len(pdb_files)} PDB entries, successfully linked/copied {linked_count}")
            print(f"Linked {len(pdb_files)} PDB files to custom dataset directory")
            
            # Generate features for the custom PDB files
            feature_dst = os.path.join(custom_dataset_dir, 'custom_feature.csv')
            print(f"Generating features for {len(pdb_files)} PDB files...")
            
            # Run feature generation
            cmd = [
                'python', os.path.join(protsolm_dir, 'get_feature.py'),
                '--pdb_dir', pdb_dir,
                '--out_file', feature_dst,
                '--num_workers', '4'  # Using fewer workers to avoid memory issues
            ]
            
            try:
                subprocess.run(cmd, check=True, cwd=protsolm_dir)
                print(f"Features successfully generated and saved to {feature_dst}")
            except subprocess.CalledProcessError as e:
                print(f"Error generating features: {e}")
                # If feature generation fails, use the default feature file as fallback
                feature_src = os.path.join(protsolm_dir, 'data', 'PDBSol', 'PDBSol_feature.csv')
                shutil.copy(feature_src, feature_dst)
                print(f"Falling back to copied feature file from {feature_src} to {feature_dst}")
        except Exception as e:
            print(f"Error processing PDB files: {e}")
    else:
        # If no structures provided, use a default feature file as fallback
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
        
        # Check if feature file exists, is not empty, and is valid
        feature_file_valid = False
        if os.path.exists(feature_file) and os.path.getsize(feature_file) > 0:
            # Try to read the file with pandas to ensure it's a valid CSV
            try:
                import pandas as pd
                test_df = pd.read_csv(feature_file)
                if len(test_df) > 0 and 'protein name' in test_df.columns:
                    feature_file_valid = True
                else:
                    print(f"Warning: Feature file {feature_file} exists but has no data or missing columns.")
            except Exception as e:
                print(f"Warning: Feature file {feature_file} exists but is not a valid CSV: {e}")
        else:
            print(f"Warning: Feature file {feature_file} is missing or empty.")
            
        # Create a fallback feature file if feature extraction fails
        if not feature_file_valid:
            logging.warning(f"========================= WARNING ==========================")
            logging.warning(f"Feature file validation failed. Creating FALLBACK feature file...")
            logging.warning(f"This means real features could not be extracted from the PDB files")
            logging.warning(f"Predictions made with fallback features may be less accurate")
            logging.warning(f"========================= WARNING ==========================")
            create_fallback_feature_file(abs_input_csv, feature_file)
            
            # Verify the fallback feature file was created properly
            if not validate_feature_file(feature_file):
                raise FileNotFoundError(f"Failed to create valid fallback feature file.")
            else:
                logging.warning(f"========================= NOTICE ==========================")
                logging.warning(f"FALLBACK feature file created and validated successfully")
                logging.warning(f"Predictions will be made with zeros/defaults for all features")
                logging.warning(f"========================= NOTICE ===========================")
        
        # Find the model file in the ckpt directory
        model_path = os.path.join(protsolm_dir, 'ckpt', 'feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        print(f"Using model file: {model_path}")
        
        # Run the command with all required parameters per README
        cmd = [
            'python', script_name,
            '--supv_dataset', custom_dataset_dir,
            '--test_file', abs_input_csv,
            '--test_result_dir', abs_output_dir,
            '--feature_file', feature_file,
            '--gnn_model_path', model_path,
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
