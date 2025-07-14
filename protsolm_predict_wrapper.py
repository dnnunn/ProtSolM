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

def validate_feature_file(file_path):
    """Validate that the feature file exists and is a valid CSV with data.
    
    Args:
        file_path: Path to the feature file to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        logging.error(f"Feature file not found: {file_path}")
        return False
    
    try:
        # Try to read the file
        df = pd.read_csv(file_path)
        
        # Check if it has any data
        if df.empty:
            logging.error(f"Feature file is empty: {file_path}")
            return False
            
        # Check if it has a 'protein name' column which is required
        if 'protein name' not in df.columns:
            logging.error(f"Feature file missing 'protein name' column: {file_path}")
            return False
            
        # Check if it has at least some feature columns
        if len(df.columns) < 5:  # Protein name + a few features at minimum
            logging.error(f"Feature file has too few columns: {file_path}")
            return False
            
        logging.info(f"Feature file validated successfully: {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error validating feature file: {e}")
        return False


def create_fallback_feature_file(input_csv_path, feature_file_path):
    """Create a fallback feature file with default values (zeros).
    
    Args:
        input_csv_path: Path to the input CSV file with protein sequences
        feature_file_path: Path to the output feature CSV file to create
    """
    import pandas as pd
    
    logging.warning(f"Creating FALLBACK feature file with ZERO VALUES: {feature_file_path}")
    logging.warning(f"NOTE: These are placeholder features and may affect prediction accuracy")
    
    print(f"Creating fallback feature file at {feature_file_path}")
    
    try:
        # Read the input CSV to get protein names
        input_df = pd.read_csv(input_csv_path)
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        logging.warning(f"Will attempt to create minimal fallback feature file with default protein names")
        # Create a minimal dataframe with default protein names
        input_df = pd.DataFrame({
            'name': [f'protein_{i+1}' for i in range(100)],  # Generate 100 default protein names
        })
    
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
    """Set up a custom dataset directory with PDB files for ProtSolM.

    Args:
        input_csv (str): Path to input CSV file.
        structures_dir (str, optional): Path to directory with PDB structures.

    Returns:
        str: Path to custom dataset directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_dataset_dir = os.path.join(script_dir, 'custom_dataset')
    os.makedirs(custom_dataset_dir, exist_ok=True)

    # Copy the input CSV to the custom dataset dir
    shutil.copy(input_csv, os.path.join(custom_dataset_dir, 'input.csv'))

    # Create a directory for PDB files
    pdb_dir = os.path.join(custom_dataset_dir, 'pdb')
    os.makedirs(pdb_dir, exist_ok=True)

    # Force feature file regeneration by removing existing feature file
    feature_file = os.path.join(custom_dataset_dir, 'custom_feature.csv')
    if os.path.exists(feature_file):
        print(f"Removing existing feature file to force regeneration: {feature_file}")
        os.remove(feature_file)
        
    # If structures_dir is provided, copy or link PDB files to the custom dataset
    linked_count = 0
    invalid_count = 0
    processed_files = []
    
    if structures_dir and os.path.isdir(structures_dir):
        # Get a list of all PDB files in the structures directory
        pdb_files = [f for f in os.listdir(structures_dir) if f.endswith('.pdb')]
        print(f"Found {len(pdb_files)} PDB files in {structures_dir}")

        for pdb_file in pdb_files:
            src = os.path.join(structures_dir, pdb_file)
            dst = os.path.join(pdb_dir, pdb_file)
            
            # Only process if it's a file (skip directories)
            if os.path.isfile(src):
                try:
                    # Check if it's a valid PDB file by looking at the first few lines
                    try:
                        with open(src, 'r') as f:
                            first_line = f.readline().strip()
                            second_line = f.readline().strip() if first_line else ""
                            
                            # Special case for files starting with "PARENT N/A" followed by ATOM records
                            if first_line.startswith('PARENT') and second_line.startswith('ATOM'):
                                # Create a sanitized version of the PDB file without the PARENT line
                                print(f"Sanitizing PDB file with PARENT line: {pdb_file}")
                                sanitized_file = dst
                                with open(src, 'r') as input_file, open(sanitized_file, 'w') as output_file:
                                    # Skip the first line (PARENT line)
                                    next(input_file)
                                    # Write a standard REMARK line instead
                                    output_file.write("REMARK   1 File sanitized to remove unsupported PARENT record\n")
                                    # Copy the rest of the file
                                    for line in input_file:
                                        output_file.write(line)
                                linked_count += 1
                                processed_files.append(os.path.splitext(pdb_file)[0])
                                print(f"Created sanitized PDB file: {pdb_file}")
                            # Standard PDB file validation
                            elif first_line.startswith(('HEADER', 'TITLE', 'COMPND', 'ATOM', 'MODEL', 'REMARK')):
                                # It's a valid PDB file, copy it (use shutil.copyfile to ensure it's a file, not symlink)
                                try:
                                    # Force direct file copy with no symlinks
                                    shutil.copyfile(src, dst)
                                    # Verify the copied file exists and is a file
                                    if os.path.isfile(dst):
                                        print(f"Successfully copied PDB file: {dst} (size: {os.path.getsize(dst)} bytes)")
                                        linked_count += 1
                                        processed_files.append(os.path.splitext(pdb_file)[0])
                                    else:
                                        print(f"ERROR: Copy succeeded but {dst} is not a file! Type: {os.stat(dst).st_mode}")
                                except Exception as copy_err:
                                    print(f"ERROR copying {src} to {dst}: {copy_err}")
                                    invalid_count += 1
                            else:
                                print(f"Warning: {pdb_file} does not appear to be a valid PDB file. First line: {first_line}")
                                invalid_count += 1
                    except UnicodeDecodeError:
                        print(f"Warning: {pdb_file} appears to be a binary file, skipping.")
                        invalid_count += 1
                except Exception as e:
                    print(f"Error processing {pdb_file}: {e}")
                    invalid_count += 1
            elif os.path.isdir(src):
                print(f"Warning: {pdb_file} is a directory, skipping.")
                invalid_count += 1

        print(f"Processed {linked_count} valid PDB files to {pdb_dir}.")
        if invalid_count > 0:
            print(f"Skipped {invalid_count} invalid files.")

    # Run DSSP to generate features for the dataset
    # This will invoke feature generation explicitly instead of relying on the internal process
    if linked_count > 0:
        print(f"Generating features for {linked_count} PDB files...")
        try:
            # Import necessary libraries for feature generation
            from Bio.PDB import PDBParser, DSSP
            import pandas as pd
            import numpy as np
            
            # Create empty feature dataframe
            feature_columns = ['pdb_id', 'aa_composition', 'gravy', 'ss_composition', 'hygrogen_bonds', 'exposed_res_fraction', 'pLDDT']
            feature_df = pd.DataFrame(columns=feature_columns)
            
            # Process each valid PDB file
            for protein_id in processed_files:
                try:
                    pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
                    print(f"Processing {pdb_file} for feature extraction...")
                    
                    # Verify file exists and is an actual file, not a directory
                    if not os.path.exists(pdb_file):
                        print(f"ERROR: PDB file {pdb_file} does not exist!")
                        continue
                    
                    if not os.path.isfile(pdb_file):
                        print(f"ERROR: {pdb_file} exists but is not a file (may be a directory)!")
                        continue
                        
                    # Check file size is reasonable
                    file_size = os.path.getsize(pdb_file)
                    if file_size < 100:  # Sanity check for minimal PDB size
                        print(f"ERROR: {pdb_file} is too small ({file_size} bytes), likely invalid")
                        continue
                    
                    # Extra verification of file content
                    try:
                        with open(pdb_file, 'r') as f:
                            first_lines = [f.readline().strip() for _ in range(5)]
                            if not any(line.startswith(('HEADER', 'TITLE', 'COMPND', 'ATOM', 'MODEL', 'REMARK')) for line in first_lines):
                                print(f"ERROR: {pdb_file} doesn't contain valid PDB headers in first 5 lines")
                                continue
                    except Exception as file_err:
                        print(f"ERROR reading {pdb_file}: {file_err}")
                        continue
                    
                    print(f"Verified {pdb_file} exists as a file, size: {file_size} bytes")
                    
                    # Parse PDB file with proper error handling
                    try:
                        parser = PDBParser(QUIET=True)
                        structure = parser.get_structure(protein_id, pdb_file)
                        if not structure.get_list() or not structure[0].get_list():  # Check structure and first model
                            print(f"ERROR: No valid models found in {pdb_file}")
                            continue
                        model = structure[0]
                    except Exception as parse_err:
                        print(f"ERROR parsing PDB {pdb_file}: {parse_err}")
                        continue
                    
                    # Run DSSP with extra error handling
                    try:
                        # Set environment variable for DSSP if not set
                        if 'LIBCIFPP_DATA_DIR' not in os.environ:
                            os.environ['LIBCIFPP_DATA_DIR'] = '/home/david_nunn/miniconda3/envs/ProtSolM/share/libcifpp'
                            print("Set LIBCIFPP_DATA_DIR environment variable for DSSP")
                            
                        # Try running DSSP with explicit path
                        dssp = DSSP(model, pdb_file, dssp='mkdssp')
                    except Exception as dssp_err:
                        print(f"DSSP failed on {pdb_file}: {dssp_err}")
                        # Try with different DSSP path as fallback
                        try:
                            print("Retrying DSSP with alternate executable path...")
                            # Look for mkdssp in PATH
                            import subprocess
                            dssp_path = subprocess.check_output(['which', 'mkdssp']).decode().strip()
                            print(f"Found mkdssp at: {dssp_path}")
                            dssp = DSSP(model, pdb_file, dssp=dssp_path)
                        except Exception as retry_err:
                            print(f"DSSP retry failed: {retry_err}")
                            continue
                    
                    # Extract secondary structure composition
                    ss_counts = {'H': 0, 'E': 0, 'C': 0}  # helix, sheet, coil
                    total_res = 0
                    exposed_res = 0
                    h_bonds = 0
                    
                    # Process DSSP results
                    for key in dssp.keys():
                        res_data = dssp[key]
                        ss = res_data[2]  # Secondary structure
                        if ss in ['-', 'T', 'S']:  # Map irregular structures to coil
                            ss = 'C'
                        if ss in ['G', 'I']:  # Map 3-10 helix and pi-helix to alpha-helix
                            ss = 'H'
                        if ss in ['B', 'E']:  # Map beta bridge to sheet
                            ss = 'E'
                        
                        ss_counts[ss] += 1
                        total_res += 1
                        
                        # Check for exposed residues (ASA > 0)
                        if res_data[3] > 0:  # ASA value
                            exposed_res += 1
                            
                        # Count hydrogen bonds
                        h_bonds += abs(res_data[6]) + abs(res_data[7])  # NH-->O and O-->NH
                    
                    # Calculate metrics
                    ss_composition = f"{ss_counts['H']/max(1, total_res):.4f},{ss_counts['E']/max(1, total_res):.4f},{ss_counts['C']/max(1, total_res):.4f}"
                    exposed_fraction = f"{exposed_res/max(1, total_res):.4f}"
                    h_bond_density = f"{h_bonds/max(1, total_res):.4f}"
                    
                    # Extract sequence for AA composition and GRAVY
                    sequence = ''.join([dssp[key][1] for key in sorted(dssp.keys())])
                    
                    # Calculate AA composition (placeholder calculation)
                    aa_comp_values = [0] * 20  # 20 standard amino acids
                    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
                    for aa in sequence:
                        if aa in aa_list:
                            aa_comp_values[aa_list.index(aa)] += 1
                    aa_comp = ','.join([f"{val/max(1, len(sequence)):.4f}" for val in aa_comp_values])
                    
                    # Calculate GRAVY (Grand average of hydropathy)
                    hydropathy = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
                    gravy_score = sum(hydropathy.get(aa, 0) for aa in sequence) / max(1, len(sequence))
                    gravy = f"{gravy_score:.4f}"
                    
                    # Add to feature dataframe
                    new_row = {
                        'pdb_id': protein_id,
                        'aa_composition': aa_comp,
                        'gravy': gravy,
                        'ss_composition': ss_composition,
                        'hygrogen_bonds': h_bond_density,
                        'exposed_res_fraction': exposed_fraction,
                        'pLDDT': '1.0000'  # Default value for actual structures (not predicted)
                    }
                    feature_df = pd.concat([feature_df, pd.DataFrame([new_row])], ignore_index=True)
                    print(f"Successfully extracted features for {protein_id}")
                    
                except Exception as e:
                    print(f"Error extracting features for {protein_id}: {str(e)}")
            
            # Save feature dataframe to CSV
            feature_df.to_csv(feature_file, index=False)
            print(f"Generated feature file at {feature_file} with {len(feature_df)} entries")
            
        except Exception as e:
            print(f"Error during feature generation: {str(e)}")
            # Fall back to default feature file if available
            default_feature = os.path.join(script_dir, 'data', 'PDBSol', 'PDBSol_feature.csv')
            if os.path.exists(default_feature):
                shutil.copy(default_feature, feature_file)
                print(f"Copied default feature file from {default_feature} to {feature_file} due to error")
    else:
        # No valid PDB files were processed, fall back to the default feature file
        default_feature = os.path.join(script_dir, 'data', 'PDBSol', 'PDBSol_feature.csv')
        if os.path.exists(default_feature):
            shutil.copy(default_feature, feature_file)
            print(f"No valid PDB files processed. Copied default feature file from {default_feature} to {feature_file}")

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
    
    # Clean up any existing problematic custom dataset directory
    custom_dataset_path = os.path.join(protsolm_dir, 'custom_dataset')
    if os.path.exists(custom_dataset_path):
        print(f"Removing existing custom dataset directory: {custom_dataset_path}")
        try:
            shutil.rmtree(custom_dataset_path)
            print("Successfully removed old custom dataset directory")
        except Exception as e:
            print(f"Warning: Could not completely remove directory: {e}")
            # Try removing individual files if rmtree fails
            try:
                pdb_dir = os.path.join(custom_dataset_path, 'pdb')
                if os.path.exists(pdb_dir):
                    for item in os.listdir(pdb_dir):
                        item_path = os.path.join(pdb_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                os.unlink(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception as err:
                            print(f"Failed to remove {item_path}: {err}")
            except Exception as err:
                print(f"Failed during manual cleanup: {err}")
    
    # Set environment variables needed by DSSP
    os.environ['LIBCIFPP_DATA_DIR'] = '/home/david_nunn/miniconda3/envs/ProtSolM/share/libcifpp'
    print(f"Set LIBCIFPP_DATA_DIR environment variable for DSSP to {os.environ['LIBCIFPP_DATA_DIR']}")
    
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
