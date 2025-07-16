#!/usr/bin/env python3
"""
Convert FASTA file to CSV format for ProtSolM.
"""
import os
import sys
import pandas as pd
from Bio import SeqIO

def fasta_to_csv(fasta_file, csv_file, id_list_file=None):
    """Convert a FASTA file to CSV format with protein_name and sequence columns. Optionally write canonical ID list."""
    proteins = []
    id_list = []
    
    # Parse the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        sequence = str(record.seq)
        proteins.append({
            'id': protein_id,
            'aa_seq': sequence,
            'label': 0
        })
        id_list.append(protein_id)
    df = pd.DataFrame(proteins)
    df.to_csv(csv_file, index=False)
    if id_list_file:
        with open(id_list_file, 'w') as f:
            for pid in id_list:
                f.write(f"{pid}\n")
        print(f"Wrote canonical ID list to {id_list_file}")
    print(f"Converted {len(proteins)} sequences from {fasta_file} to {csv_file}")
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <fasta_file> <output_csv> [output_id_list.txt]")
        sys.exit(1)
    fasta_file = sys.argv[1]
    csv_file = sys.argv[2]
    id_list_file = sys.argv[3] if len(sys.argv) > 3 else None
    fasta_to_csv(fasta_file, csv_file, id_list_file)
