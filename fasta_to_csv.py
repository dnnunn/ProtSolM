#!/usr/bin/env python3
"""
Convert FASTA file to CSV format for ProtSolM.
"""
import os
import sys
import pandas as pd
from Bio import SeqIO

def fasta_to_csv(fasta_file, csv_file):
    """Convert a FASTA file to CSV format with protein_name and sequence columns."""
    proteins = []
    
    # Parse the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract protein ID from the FASTA header
        protein_id = record.id
        sequence = str(record.seq)
        
        proteins.append({
            'id': protein_id,
            'aa_seq': sequence,
            'label': 0  # Placeholder for evaluation
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(proteins)
    df.to_csv(csv_file, index=False)
    print(f"Converted {len(proteins)} sequences from {fasta_file} to {csv_file}")
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <fasta_file> <output_csv>")
        sys.exit(1)
        
    fasta_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    fasta_to_csv(fasta_file, csv_file)
