import os
import sys
import argparse
import mdtraj as md
import pandas as pd
import biotite.structure.io as bsio
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
    """If first line is not HEADER or TITLE, insert HEADER and use a temp file."""
    import tempfile
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    if lines and not (lines[0].startswith('HEADER') or lines[0].startswith('TITLE')):
        tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.pdb')
        tmp.write('HEADER    GENERATED FOR DSSP\n')
        tmp.writelines(lines)
        tmp.close()
        return tmp.name, True
    return pdb_file, False

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
            import os
            os.unlink(pdb_for_dssp)

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
