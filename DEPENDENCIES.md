# ProtSolM Dependencies

## Required Packages

Beyond the packages specified in `environment.yaml`, the following dependencies are required:

### DSSP (Dictionary of Protein Secondary Structure)

DSSP is required for feature extraction from PDB files. It's used to compute secondary structure and solvent accessibility.

#### Installation

Install DSSP using conda-forge (recommended method):

```bash
conda activate ProtSolM
conda install -c conda-forge dssp
```

Verify installation:
```bash
which mkdssp
# Should output the path to the mkdssp executable
```

If the installation fails, you may see errors like:
```
[Errno 2] No such file or directory: 'mkdssp'
```

### Other Dependencies

The feature extraction pipeline also requires:
- mdtraj (for hydrogen bond detection)
- biotite (for structure parsing and pLDDT extraction)
- BioPython (for parsing PDB files)

These are typically included in the conda environment.

## Troubleshooting

If feature generation fails with missing DSSP errors, check:

1. That DSSP is properly installed (`which mkdssp` should return a path)
2. That PDB files are valid and properly formatted
3. That the conda environment is activated before running scripts
