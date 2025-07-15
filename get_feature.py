import os
import sys
import argparse
import mdtraj as md
import pandas as pd
import biotite.structure.io as bsio
import re
import tempfile
import logging
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
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
    Builds a clean PDB file from scratch to ensure DSSP compatibility.

    This function reads an input PDB, writes canonical HEADER, TITLE, and CRYST1
    records, then processes and appends only ATOM/HETATM records, ensuring they
    are correctly formatted. It finishes by adding TER and END records.
    This ground-up approach is more robust than patching existing lines.
    """
    output_lines = []

    # 1. Add canonical HEADER, TITLE, and CRYST1 records (80 chars)
    header = 'HEADER    PEPTIDE PROJECT PDB                01-JAN-00   XXXX'
    title = f'TITLE     {os.path.splitext(os.path.basename(pdb_file))[0]}'
    cryst1 = 'CRYST1   40.960   18.650   22.520  90.00  90.77  90.00 P 1           1'
    output_lines.append(header.ljust(80) + '\n')
    output_lines.append(title.ljust(80) + '\n')
    output_lines.append(cryst1.ljust(80) + '\n')

    last_atom_line_info = None
    atom_serial_counter = 0

    # 2. Process only ATOM/HETATM records from the original file
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_serial_counter += 1
                # Ensure chain ID in column 22 is 'A' if blank
                if len(line) >= 22 and line[21] == ' ':
                    line = line[:21] + 'A' + line[22:]

                # The most robust fix: use string slicing for the fixed-width PDB format.
                # This avoids all regex complexities and guarantees correct spacing.
                record_type = line[0:6]
                atom_serial = line[6:11]
                space1 = ' '
                atom_name = line[12:16].strip()
                rest_of_line = line[16:]

                # Reconstruct the line with guaranteed correct PDB formatting
                fixed_line = f"{record_type}{atom_serial}{space1}{atom_name.ljust(4)}{rest_of_line}".rstrip() 

                output_lines.append(fixed_line.ljust(80) + '\n')
                
                # Store info from the last atom for the TER record
                last_atom_line_info = {
                    'resName': line[17:20].strip(),
                    'chainID': line[21:22].strip() if line[21:22].strip() else 'A',
                    'resSeq': line[22:26].strip()
                }

    # 3. Add a canonical TER record
    if last_atom_line_info:
        ter_res_name = last_atom_line_info['resName']
        ter_chain_id = last_atom_line_info['chainID']
        ter_res_seq = last_atom_line_info['resSeq']
        ter_serial = atom_serial_counter + 1
        ter_record = f'TER   {ter_serial:>5}      {ter_res_name:>3} {ter_chain_id}{ter_res_seq:>4}'
        output_lines.append(ter_record.ljust(80) + '\n')

    # 4. Add a canonical END record
    output_lines.append('END'.ljust(80) + '\n')

    # 5. Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdb', dir='/var/tmp') as tmp_file:
        tmp_file.writelines(output_lines)
        sanitized_pdb_path = tmp_file.name
    
    logging.debug(f"Created sanitized PDB (ground-up): {sanitized_pdb_path}")
    return sanitized_pdb_path

def generate_feature(pdb_file):
    sanitized_pdb_path = None
    try:
        # extract amino acid sequence
        aa_seq = extract_seq_from_pdb(pdb_file)
        sanitized_pdb_path = sanitize_pdb_for_dssp(pdb_file)
        is_temp = True

        logging.debug(f"Bypassing Bio.PDB.DSSP, using direct mkdssp call on {sanitized_pdb_path}")
        
        # Primary method: direct subprocess call to mkdssp
        import tempfile
        import subprocess
        from Bio.PDB.DSSP import make_dssp_dict

        tmp_dssp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.dssp')
        tmp_dssp_out.close()
        mkdssp_cmd = ['mkdssp', sanitized_pdb_path, tmp_dssp_out.name]
        logging.debug(f"Running command: {' '.join(mkdssp_cmd)}")

        try:
            result = subprocess.run(mkdssp_cmd, capture_output=True, text=True, check=True)
            dssp_dict, _ = make_dssp_dict(tmp_dssp_out.name)
            # Convert dict to list of tuples for compatibility with downstream code
            dssp = [
                (k, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12])
                for k, d in dssp_dict.items()
            ]
            if not dssp:
                 raise ValueError("DSSP output is empty after parsing.")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            error_msg = f"Direct mkdssp call failed. File: {sanitized_pdb_path}."
            if isinstance(e, subprocess.CalledProcessError):
                error_msg += f"\nReturn Code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            else:
                error_msg += f"\nException: {e}"
            return pdb_file, error_msg
        finally:
            os.unlink(tmp_dssp_out.name)
        traj = md.load(sanitized_pdb_path)
        hbonds = md.kabsch_sander(traj)

        sec_structures = []
        rsa = []
        for i, dssp_res in enumerate(dssp):
            sec_structures.append(dssp_res[2])
            rsa.append(dssp_res[3])

    except Exception as e:
        return pdb_file, str(e) + ' (file: ' + sanitized_pdb_path + ')'
    finally:
        if 'is_temp' in locals() and is_temp:
            print(f"[DEBUG] Not deleting temp sanitized PDB: {sanitized_pdb_path}")
            # os.unlink(sanitized_pdb_path)  # Commented out for debugging

    sec_structure_str_8 = ''.join(sec_structures)
    sec_structure_str_8 = sec_structure_str_8.replace('-', 'L')
    if len(aa_seq) != len(sec_structure_str_8):
        return pdb_file, f"aa_seq {len(aa_seq)} and sec_structure_str_8 {len(sec_structure_str_8)} length mismatch"

    sec_structure_str_3 = ''.join([ss_alphabet_dic[ss] for ss in sec_structures])

    final_feature = {}
    final_feature["name"] = pdb_file.split('/')[-1]
    final_feature["aa_seq"] = aa_seq
    final_feature["ss8_seq"] = sec_structure_str_8
    final_feature["ss3_seq"] = sec_structure_str_3
    final_feature["rsa"] = rsa
    final_feature["hbonds_num"] = hbonds[0].nnz
    
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
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
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    property_dict = {}
    if args.pdb_file:
        pdb_files = [os.path.basename(args.pdb_file)]
        pdb_dir = os.path.dirname(args.pdb_file) or '.'
    else:
        all_entries = os.listdir(args.pdb_dir)
        pdb_files = [f for f in all_entries if os.path.isfile(os.path.join(args.pdb_dir, f)) and f.endswith('.pdb')]
        pdb_dir = args.pdb_dir
    print("PDB files to process:", pdb_files)
    for f in pdb_files:
        full_path = os.path.join(pdb_dir, f)
        print(f"{f}: isfile={os.path.isfile(full_path)}, size={os.path.getsize(full_path)}")
    
    def process_pdb(pdb_file, pdb_dir):
        features, error = generate_feature(os.path.join(pdb_dir, pdb_file))
        if error or not isinstance(features, dict):
            print(f"Skipping {pdb_file}: error={error}, features type={type(features)}")
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
