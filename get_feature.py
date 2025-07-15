import os
import sys
import argparse
import mdtraj as md
import pandas as pd
import biotite.structure.io as bsio
import re
import tempfile
import logging
import subprocess
import biotite.structure.io as strucio
sys.path.append(os.getcwd())

# Set up logging
log_file = 'feature_extraction.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

from Bio import PDB
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.data_utils import extract_seq_from_pdb

ss_alphabet = ['H', 'E', 'C']
ss_alphabet_dic = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C"
}


def sanitize_pdb_for_dssp(pdb_file):
    """
    Sanitizes a PDB file for DSSP by ensuring atom serial numbers are sequential and adding a default CRYST1 record if missing.
    This is a minimal-touch approach that preserves original formatting.
    """
    sanitized_lines = []
    atom_counter = 1
    
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    # Add CRYST1 record if missing to handle DSSP parsing errors
    if not any(line.startswith('CRYST1') for line in lines):
        cryst1_line = "CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           1          \n"
        lines.insert(0, cryst1_line)
    
    # Check if CRYST1 record exists; if not, add a default one
    has_cryst1 = any(line.startswith('CRYST1') for line in lines)
    if not has_cryst1:
        sanitized_lines.append('CRYST1    1.000    1.000    1.000  90.00  90.00 90.00 P 1           1\n')
        logging.debug("Added default CRYST1 record as it was missing.")
    
    for line in lines:
        if line.startswith(('ATOM', 'HETATM', 'TER')):
            # Re-number the serial number in place for ATOM/HETATM/TER records
            new_serial = str(atom_counter).rjust(5)
            modified_line = line[:6] + new_serial + line[11:]
            sanitized_lines.append(modified_line)
            atom_counter += 1
        else:
            # Keep all other lines unchanged
            sanitized_lines.append(line)
    
    # Ensure there's an END record if one doesn't exist
    if not any(line.startswith('END') for line in sanitized_lines):
        sanitized_lines.append('END\n')
    
    temp_pdb = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdb', dir='/var/tmp')
    temp_pdb.writelines(sanitized_lines)
    temp_pdb.close()
    logging.debug(f"Created sanitized PDB (with CRYST1 check): {temp_pdb.name}")
    return temp_pdb.name

def custom_dssp_parser(dssp_file):
    """
    Parses a DSSP output file manually to avoid Bio.PDB parser issues.
    Returns a dictionary with the same structure as the one from Bio.PDB.DSSP.
    """
    dssp_dict = {}
    with open(dssp_file, 'r') as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if "  #  RESIDUE" in line:
            start_idx = i + 1
            break

    if start_idx == 0:
        return dssp_dict

    for line in lines[start_idx:]:
        if len(line) > 38 and line[13] != '!' and line[13].strip():
            try:
                res_num = int(line[5:10].strip())
                chain_id = line[11].strip() or 'A'
                aa = line[13]
                ss = line[16]
                if ss == ' ': ss = 'C'  # Map blank to Coil
                rsa = float(line[35:38].strip())

                dict_key = (chain_id, (' ', res_num, ' '))
                # Value: (amino_acid, sec_structure, rsa, ... other fields ...)
                # We only need the first 3 for our features.
                dssp_dict[dict_key] = (aa, ss, rsa)
            except (ValueError, IndexError):
                continue
    return dssp_dict

def generate_feature(pdb_file):
    try:
        # extract amino acid sequence
        aa_seq = extract_seq_from_pdb(pdb_file)

        logging.debug(f"Using direct mkdssp call on {pdb_file}")
        
        # Primary method: direct subprocess call to mkdssp
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dssp', dir='/var/tmp') as dssp_out:
            dssp_out_name = dssp_out.name
        
        command = ['mkdssp', pdb_file, dssp_out_name]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            error_message = f"mkdssp failed for {os.path.basename(pdb_file)} with exit code {result.returncode}. Stderr: {result.stderr.strip()}"
            logging.error(error_message)
            os.remove(dssp_out_name)
            return None, result.stderr.strip()
        # Use the robust custom parser instead of BioPython's
        dssp_dict = custom_dssp_parser(dssp_out_name)
        if not dssp_dict:
             raise ValueError("DSSP output is empty after parsing.")
        # If successful, clean up the temp file
        # Clean up dssp file if it exists
        if 'dssp_out_name' in locals() and os.path.exists(dssp_out_name):
            os.remove(dssp_out_name)
        # Process the DSSP and hydrogen bond features
        traj = md.load(pdb_file)
        hbonds = md.kabsch_sander(traj)

        # Sort dictionary by residue index to ensure correct order
        sorted_keys = sorted(dssp_dict.keys(), key=lambda k: k[1][1])
        
        sec_structures = [dssp_dict[key][1] for key in sorted_keys]
        # RSA from custom parser is already a float, no need for dssp_dict[key][2]
        rsa = [dssp_dict[key][2] for key in sorted_keys]
    except Exception as e:
        return pdb_file, str(e)

    sec_structure_str_8 = ''.join(sec_structures)
    sec_structure_str_8 = sec_structure_str_8.replace('-', 'L')
    if len(aa_seq) != len(sec_structure_str_8):
        return pdb_file, f"aa_seq {len(aa_seq)} and sec_structure_str_8 {len(sec_structure_str_8)} length mismatch"

    sec_structure_str_3 = ''.join([ss_alphabet_dic.get(ss, 'C') for ss in sec_structures])

    final_feature = {}
    final_feature["name"] = pdb_file.split('/')[-1]
    final_feature["aa_seq"] = aa_seq
    final_feature["ss8_seq"] = sec_structure_str_8
    final_feature["ss3_seq"] = sec_structure_str_3
    final_feature["rsa"] = rsa
    final_feature["hbonds_num"] = hbonds[0].nnz
    
    struct = strucio.load_structure(pdb_file, extra_fields=["b_factor"])
    final_feature["pLDDT"] = struct.b_factor.mean()

    return final_feature, None


def properties_from_sequence(features):
    sequence = features["aa_seq"]
    length = len(sequence)
    counts = {
        'C': sequence.count('C'),
        'D': sequence.count('D'),
        'E': sequence.count('E'),
        'R': sequence.count('R'),
        'H': sequence.count('H'),
        'N': sequence.count('N'),
        'G': sequence.count('G'),
        'P': sequence.count('P'),
        'S': sequence.count('S')
    }

    amino_acid_hydropathy = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
        'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }

    total_hydropathy = sum(amino_acid_hydropathy.get(aa, 0) for aa in sequence)
    average_hydropathy = total_hydropathy / len(sequence)

    properties = {
        'L': length,
        '1-C': counts['C'] / length,
        '1-D': counts['D'] / length,
        '1-E': counts['E'] / length,
        '1-R': counts['R'] / length,
        '1-H': counts['H'] / length,
        'Turn-forming residues fraction': (counts['N'] + counts['G'] + counts['P'] + counts['S']) / length,
        'GRAVY': average_hydropathy
    }

    return properties


def properties_from_dssp(features):
    ss8_seq = features["ss8_seq"]
    ss3_seq = features["ss3_seq"]
    rsa = features["rsa"]
    total_residues = len(ss8_seq)
    name = features["name"]
    h_num = features["hbonds_num"]
    plddt = features["pLDDT"]

    ss8_counts = {"G": 0, "H": 0, "I": 0, "B": 0, "E": 0, "T": 0, "S": 0, "P": 0, "L": 0}
    ss3_counts = {"H": 0, "E": 0, "C": 0}
    cutoff = [x / 100 for x in range(5, 105, 5)]
    exposed_residues = [0] * 20

    for i in range(total_residues):
        ss8_counts[ss8_seq[i]] += 1
        ss3_counts[ss3_seq[i]] += 1

        for j in range(20):
            if rsa[i] >= cutoff[j]:
                exposed_residues[j] += 1

    properties = {
        "ss8-G": ss8_counts["G"] / total_residues,
        "ss8-H": ss8_counts["H"] / total_residues,
        "ss8-I": ss8_counts["I"] / total_residues,
        "ss8-B": ss8_counts["B"] / total_residues,
        "ss8-E": ss8_counts["E"] / total_residues,
        "ss8-T": ss8_counts["T"] / total_residues,
        "ss8-S": ss8_counts["S"] / total_residues,
        "ss8-P": ss8_counts["P"] / total_residues,
        "ss8-L": ss8_counts["L"] / total_residues,
        "ss3-H": ss3_counts["H"] / total_residues,
        "ss3-E": ss3_counts["E"] / total_residues,
        "ss3-C": ss3_counts["C"] / total_residues,
        "Hydrogen bonds": h_num,
        "Hydrogen bonds per 100 residues": h_num * 100 / total_residues,
        f"Exposed residues fraction by 5%": exposed_residues[0] / total_residues,
        f"Exposed residues fraction by 10%": exposed_residues[1] / total_residues,
        f"Exposed residues fraction by 15%": exposed_residues[2] / total_residues,
        f"Exposed residues fraction by 20%": exposed_residues[3] / total_residues,
        f"Exposed residues fraction by 25%": exposed_residues[4] / total_residues,
        f"Exposed residues fraction by 30%": exposed_residues[5] / total_residues,
        f"Exposed residues fraction by 35%": exposed_residues[6] / total_residues,
        f"Exposed residues fraction by 40%": exposed_residues[7] / total_residues,
        f"Exposed residues fraction by 45%": exposed_residues[8] / total_residues,
        f"Exposed residues fraction by 50%": exposed_residues[9] / total_residues,
        f"Exposed residues fraction by 55%": exposed_residues[10] / total_residues,
        f"Exposed residues fraction by 60%": exposed_residues[11] / total_residues,
        f"Exposed residues fraction by 65%": exposed_residues[12] / total_residues,
        f"Exposed residues fraction by 70%": exposed_residues[13] / total_residues,
        f"Exposed residues fraction by 75%": exposed_residues[14] / total_residues,
        f"Exposed residues fraction by 80%": exposed_residues[15] / total_residues,
        f"Exposed residues fraction by 85%": exposed_residues[16] / total_residues,
        f"Exposed residues fraction by 90%": exposed_residues[17] / total_residues,
        f"Exposed residues fraction by 95%": exposed_residues[18] / total_residues,
        f"Exposed residues fraction by 100%": exposed_residues[19] / total_residues,
        "pLDDT": plddt,
        "protein name": name
    }

    return properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--pdb_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--sanitize', type=str, help='Path to PDB file to sanitize for DSSP compatibility')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    property_dict = {}
    if args.pdb_file:
        pdb_files = [os.path.basename(args.pdb_file)]
        pdb_dir = os.path.dirname(args.pdb_file) or '.'
    else:
        pdb_dir = args.pdb_dir
        # Run the formatting script as a subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'format_pdb.sh')
        if os.path.exists(script_path):
            logging.info(f"Running PDB formatting script in: {pdb_dir}")
            subprocess.run(['bash', script_path], cwd=pdb_dir, check=True)
            # Update pdb_dir to point to the directory with formatted files
            pdb_dir = os.path.join(pdb_dir, 'pdb_final')
            logging.info(f"Using formatted PDBs from: {pdb_dir}")
        else:
            logging.warning(f"PDB formatting script not found at {script_path}. Using original PDBs.")

        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    logging.info(f"{len(pdb_files)} PDB files to process.")
    
    if args.sanitize:
        sanitized_path = sanitize_pdb_for_dssp(args.sanitize)
        print(f"Sanitized PDB file created at: {sanitized_path}")
        sys.exit(0)

    def process_pdb(pdb_file, pdb_dir):
        features, error = generate_feature(os.path.join(pdb_dir, pdb_file))
        if error or not isinstance(features, dict):
            logging.warning(f"Skipping {pdb_file} due to processing error: {error}")
            return None
        properties_seq = properties_from_sequence(features)
        properties_dssp = properties_from_dssp(features)
        properties = {}
        properties.update(properties_seq)
        properties.update(properties_dssp)
        return properties

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_pdb, pdb, pdb_dir) for pdb in pdb_files]
        for future in tqdm(as_completed(futures), total=len(pdb_files)):
            properties = future.result()
            if properties is not None:
                for k, v in properties.items():
                    if k not in property_dict:
                        property_dict[k] = []
                    property_dict[k].append(v)
    
    pd.DataFrame(property_dict).to_csv(args.out_file, index=False)
