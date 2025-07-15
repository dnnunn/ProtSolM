#!/bin/bash

# This script formats minimalist PDB files to be fully compliant with
# the strict validation checks of modern mkdssp versions.
# It adds HEADER, TITLE, COMPND, SOURCE, KEYWDS, AUTHOR, SEQRES, and CRYST1 records.

OUTPUT_DIR="pdb_final"
DATE=$(date +%d-%b-%y | tr '[:lower:]' '[:upper:]')
AUTHOR_NAME="D. NUNN" # Using your name as a placeholder

# --- Main Logic ---
echo "⚙️ Starting final batch formatting..."
mkdir -p "$OUTPUT_DIR"

# Loop through every file ending with .pdb
for pdb_file in *.pdb; do
    if [ ! -f "$pdb_file" ]; then continue; fi
    echo "Processing: ${pdb_file}"

    base_name=$(basename "$pdb_file" .pdb)
    output_file="${OUTPUT_DIR}/${base_name}_final.pdb"

    # --- Step 1: Extract sequence, chain ID, and residue count from ATOM records ---
    # This awk command correctly gets the sequence once per residue
    sequence_info=$(awk '
        /^ATOM/ && $3 == "CA" {
            # Check if we have seen this chain/residue combo before to avoid duplicates
            if (seen[$5,$6]++ == 0) {
                if (chain == "") { chain = $5 }
                sequence[count++] = $4
            }
        }
        END {
            printf "%s\n", chain
            for (i=0; i<count; ++i) printf "%s ", sequence[i]
            print ""
        }
    ' "$pdb_file")

    chain_id=$(echo "$sequence_info" | head -n 1)
    sequence=$(echo "$sequence_info" | tail -n +2)
    num_res=$(echo "$sequence" | wc -w)
    
    # Simple ID code generation
    number=$(echo "$base_name" | grep -o -E '[0-9]+' | tail -1)
    [[ -z "$number" ]] && number=1
    id_code=$(printf "AC%02d" "$number")
    
    # --- Step 2: Assemble the new, comprehensive PDB file ---
    {
        # 1. Mandatory Header Section
        printf "%-6s    %-40s%-10s%4s\n" "HEADER" "THEORETICAL MODEL" "$DATE" "$id_code"
        printf "%-6s    %s\n" "TITLE" "COMPUTATIONAL MODEL OF ${base_name}"
        printf "%-6s    %s\n" "COMPND" "MOL_ID: 1; MOLECULE: ${base_name}; CHAIN: ${chain_id};"
        printf "%-6s    %s\n" "SOURCE" "MOL_ID: 1; ORGANISM_SCIENTIFIC: SYNTHETIC;"
        printf "%-6s    %s\n" "KEYWDS" "THEORETICAL MODEL, PEPTIDE"
        printf "%-6s    %s\n" "EXPDTA" "THEORETICAL MODEL"
        printf "%-6s    %-s\n" "AUTHOR" "$AUTHOR_NAME"
        
        # 2. Add any existing remarks from the source file
        grep "^REMARK" "$pdb_file"
        
        # 3. Add generated SEQRES records
        ser_num=1
        seq_array=($sequence)
        for ((i = 0; i < num_res; i += 13)); do
            chunk=("${seq_array[@]:i:13}")
            printf "SEQRES %3d %1s %4d  %s\n" "$ser_num" "$chain_id" "$num_res" "${chunk[*]}"
            ((ser_num++))
        done
        
        # 4. Add CRYST1 record
        printf "%-6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d\n" \
        "CRYST1" 100.000 100.000 100.000 90.00 90.00 90.00 "P 1" 1
        
        # 5. Add the original ATOM/TER/END records
        grep -v "^REMARK" "$pdb_file"

    } > "$output_file"
done

echo "✅ Batch processing complete. Files are in '${OUTPUT_DIR}'."
