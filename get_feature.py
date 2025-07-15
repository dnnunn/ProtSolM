import os
import sys
import argparse
import mdtraj as md
import pandas as pd
import biotite.structure.io as bsio
import re
sys.path.append(os.getcwd())
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
    Ensures PDB file has a unique HEADER, TITLE, and CRYST1 line for DSSP compatibility.
    HEADER will be 'Peptide Project PDB', TITLE will be the PDB filename (without extension), and CRYST1 will be a minimal dummy cell.
    Returns (path_to_sanitized_pdb, is_temp_file).
    """
    import tempfile, os
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    header_inserted = False
    title_inserted = False
    cryst1_inserted = False
    # Canonical HEADER: columns 11-50 = classification, 51-59 = date, 63-66 = PDB ID
    # Example: HEADER    PEPTIDE PROJECT PDB                01-JAN-00   XXXX
    header_line = ("HEADER    PEPTIDE PROJECT PDB".ljust(50) + "01-JAN-00".rjust(9) + "   XXXX".rjust(11) + "\n")
    title_line = (f"TITLE     {os.path.splitext(os.path.basename(pdb_file))[0]}".ljust(80) + "\n")
    # Insert CRYST1 if missing (use canonical 1CRN values, exact format)
    cryst1_line = 'CRYST1   40.960   18.650   22.520  90.00  90.77  90.00 P 1           1\n'

    # Always start with HEADER, TITLE, CRYST1
    output_lines = [header_line, title_line, cryst1_line]
    for line in lines:
        if line.startswith('HEADER') or line.startswith('TITLE') or line.startswith('CRYST1'):
            continue
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Fix chain ID (column 22, 1-based) if missing or blank
            line = line.rstrip('\n')
            if len(line) >= 22 and (line[21] == ' ' or line[21] == ''):
                line = line[:21] + 'A' + line[22:]
            # Mimic sed: replace double space after atom serial with single space
            fixed_line = re.sub(r'^(ATOM  .{5})  ', r'\1 ', line)
            # Pad to 80 characters
            output_lines.append(fixed_line[:80].ljust(80) + "\n")
        else:
            # Pad END lines to 80 characters
            if line.startswith('END'):
                output_lines.append(line.rstrip().ljust(80) + "\n")
            # Pad TITLE lines to 80 characters
            elif line.startswith('TITLE'):
                output_lines.append(line.rstrip().ljust(80) + "\n")
            else:
                output_lines.append(line)


    insert_idx = 0
    if not header_inserted:
        output_lines.insert(insert_idx, header_line)
        insert_idx += 1
    if not title_inserted:
        output_lines.insert(insert_idx, title_line)
        insert_idx += 1
    if not cryst1_inserted:
        output_lines.insert(insert_idx, cryst1_line)
        insert_idx += 1

    # Ensure TER after last ATOM/HETATM and END at file end
    atom_end_idx = None
    for idx, line in enumerate(output_lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_end_idx = idx
    if atom_end_idx is not None:
        # Only add TER if not present after last ATOM/HETATM
        if atom_end_idx + 1 >= len(output_lines) or not output_lines[atom_end_idx + 1].startswith('TER'):
            output_lines.insert(atom_end_idx + 1, 'TER\n')
    # Remove any trailing blank lines
    while output_lines and output_lines[-1].strip() == '':
        output_lines.pop()
    # Ensure END record
    if not output_lines or not output_lines[-1].strip().startswith('END'):
        output_lines.append('END\n')

    tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.pdb')
    tmp.writelines(output_lines)
    tmp.close()
    print(f"[DEBUG] Created sanitized PDB: {tmp.name}")
    return tmp.name, True

def generate_feature(pdb_file):
    try:
        # extract amino acid sequence
        aa_seq = extract_seq_from_pdb(pdb_file)
        pdb_for_dssp, is_temp = sanitize_pdb_for_dssp(pdb_file)
        pdb_parser = PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("protein", pdb_for_dssp)
        model = structure[0]
        print(f"[DEBUG] DSSP input file: {pdb_for_dssp}")
        try:
            # Try Bio.PDB.DSSP with explicit mkdssp path (should use correct mkdssp invocation)
            dssp = PDB.DSSP(model, pdb_for_dssp, dssp='mkdssp')
        except Exception as dssp_exc:
            # Fallback: run mkdssp via subprocess with correct arguments, then parse output
            import tempfile
            import subprocess
            from Bio.PDB import make_dssp_dict
            tmp_dssp = tempfile.NamedTemporaryFile(delete=False)
            tmp_dssp.close()
            mkdssp_cmd = ['mkdssp', pdb_for_dssp, tmp_dssp.name]
            print(f"[DEBUG] Running mkdssp command: {' '.join(mkdssp_cmd)}")
            try:
                with open('mkdssp_debug.log', 'a') as logf:
                    result = subprocess.run(
                        mkdssp_cmd,
                        stdout=logf,
                        stderr=logf,
                        text=True
                    )
                if result.returncode != 0:
                    with open('mkdssp_debug.log', 'r') as logf:
                        log_content = logf.read()
                    raise RuntimeError(f"mkdssp failed (code {result.returncode}) on {pdb_for_dssp}. Log output:\n{log_content}")
                dssp_dict, _ = make_dssp_dict(tmp_dssp.name)
                # Convert dssp_dict to DSSP-like list for compatibility
                dssp = [
                    (None, None, d['SS'], d['ASA'])
                    for k, d in dssp_dict.items()
                ]
            except Exception as sub_exc:
                return pdb_file, f"Bio.PDB.DSSP failed: {dssp_exc}; mkdssp fallback failed: {sub_exc} (file: {pdb_for_dssp})"
            finally:
                os.unlink(tmp_dssp.name)
        traj = md.load(pdb_for_dssp)
        hbonds = md.kabsch_sander(traj)

        sec_structures = []
        rsa = []
        for i, dssp_res in enumerate(dssp):
            sec_structures.append(dssp_res[2])
            rsa.append(dssp_res[3])

    except Exception as e:
        return pdb_file, e
    finally:
        if 'is_temp' in locals() and is_temp:
            print(f"[DEBUG] Not deleting temp sanitized PDB: {pdb_for_dssp}")
            # os.unlink(pdb_for_dssp)  # Commented out for debugging

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
