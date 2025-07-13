# ProtSolM Benchmark Integration Plan

## Overview

This document outlines the plan to integrate ProtSolM with the protein solubility predictor benchmarking framework, applying the same structure preparation approach used successfully with GATSol.

## Current Status

ProtSolM requires 3D protein structures (PDB files) for prediction, similar to GATSol. However, unlike GATSol which has a structure preparation process integrated in the benchmark script, ProtSolM currently lacks this integration.

## Implementation Plan

### 1. Structure Preparation Function

Create a new function `prepare_structures_for_protsolm()` in the benchmark script that:

- Uses the ESMFold Docker container to generate PDB structures from FASTA sequences
- Follows the same pattern as `prepare_structures_for_gatsol()`
- Places generated structures in a ProtSolM-specific directory
- Runs in the ProtSolM conda environment to maintain isolation

```python
def prepare_structures_for_protsolm(sequences_dict, fasta_file, timeout=300):
    """
    Prepare PDB structure files for ProtSolM using ESMFold in Docker
    This function creates a script that runs ESMFold Docker to generate structures
    and then executes it in the ProtSolM conda environment to maintain isolation
    
    Args:
        sequences_dict: Dictionary of sequence_id -> sequence
        fasta_file: Path to the input FASTA file
        timeout: Timeout for structure preparation in seconds
        
    Returns:
        bool: True if structure preparation was successful, False otherwise
    """
    # Implementation will follow the same pattern as prepare_structures_for_gatsol
    # but with ProtSolM-specific paths and configurations
    ...
```

### 2. Wrapper Script Modifications

Update the `protsolm_predict_wrapper.py` script to:

- Accept a parameter for the directory containing PDB structures
- Check for existence of required PDB files before processing
- Provide helpful error messages when structures are missing
- Include fallback behavior for handling missing structures

### 3. Benchmark Script Updates

Modify the benchmark script to:

- Call `prepare_structures_for_protsolm()` before running ProtSolM predictions
- Skip structure preparation if the `--skip-structure-prep` flag is set
- Pass the PDB directory path to the ProtSolM wrapper
- Handle structure preparation failures appropriately

### 4. Environment Isolation

Maintain strict environment isolation by:

- Using `conda run -n ProtSolM` for all ProtSolM-related operations
- Keeping all ProtSolM dependencies within its own conda environment
- Not sharing dependencies or files between predictor environments
- Using subprocess calls with appropriate environment activation

## Technical Details

### ESMFold Docker Integration

The ESMFold Docker container will be used in the same way as with GATSol:

```
docker run --gpus all --rm \
  -v /path/to/input/fasta:/app \
  -v /path/to/output/pdb:/output \
  esmfold-gpu \
  -i /app/sequences.fasta \
  -o /output
```

### Directory Structure

```
ProtSolM/
├── structures/             # Directory for PDB structures
│   └── [protein_id].pdb    # Generated PDB files
├── protsolm_predict_wrapper.py  # Updated wrapper script
└── ...
```

### Benchmark Integration Flow

1. Benchmark script calls `prepare_structures_for_protsolm()`
2. ESMFold generates PDB files in the ProtSolM structure directory
3. Benchmark script calls `run_predictor()` for ProtSolM
4. ProtSolM wrapper uses the generated structures for prediction
5. Results are saved in standardized format for benchmarking

## Testing Plan

1. First test structure generation in isolation
2. Then test the ProtSolM wrapper with pre-generated structures
3. Finally test the full integration through the benchmark script

## Future Improvements

- Add caching of generated structures to avoid redundant work
- Implement parallel structure generation for larger datasets
- Provide options for using pre-computed structures from other sources
