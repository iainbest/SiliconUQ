# SiliconUQ

Code, instructions and data to reproduce figures in paper: "Uncertainty Quantification in Atomistic Simulations of Silicon using Interatomic Potentials", authored by I. R. Best, T. J. Sullivan, J. R. Kermode. (arXiv:2402.15419)

This is organised into an overview of the environment required to run the code, and an overview of the organisation of this repository.

## Environment & Running the code

Manifest and Project toml files included for reproducible environment.

To run scripts etc., need to:
- use Julia 1.8.3
- also required is to link this Julia to a Python environment with the following packages: 
  - numpy,
  - ASE (see https://wiki.fysik.dtu.dk/ase/install.html),
  - matscipy (https://github.com/libAtoms/matscipy , Grigorev, Petr, et al. "matscipy: materials science at the atomic scale with Python." Journal of Open Source Software 9.93 (2024).),
  - PyJulia (https://github.com/JuliaPy/pyjulia), and
  - pyjulip (see https://github.com/casv2/pyjulip for installation instructions)
- activate this same Julia environment with e.g. `using Pkg; Pkg.activate(".")` in this folder

- As a quick check to verify code is working as intended, can check bulk modulus script by doing:
   - `julia ./scripts/B_increasing_basis.jl` 
   - which should run (and overwrite) a simpler version of `./outputs/dia300.csv` (for a single potential)

## Data 
Silicon 2018 GAP database from: 

A. P. Bart ́ok, J. Kermode, N. Bernstein, and G. Cs ́anyi, “Machine Learning a General-Purpose Interatomic Potential for Silicon,” Physical Review X 8, 41048 (2018), arXiv:1805.01568

https://github.com/libAtoms/silicon-testing-framework/blob/master/models/GAP/gp_iter6_sparse9k.xml.xyz

## Code 
- Julia and Python functions, classes etc. 
- Organised into code to perform QoI evaluations, and misc functions for building / optimising ACE potentials and analysis

## Potentials
Folder for holding generated potentials. This is a mostly empty folder with only a few example potentials built within using the `generate_simple_ACE_potentials.jl` script. 

This is done purely for space-saving: larger potentials can be generated (and subsequently used) locally by modifying the aforementioned script, without storing the large .h5 files in this repository.

## Scripts 
Example scripts to generate outputs for bulk modulus, elastic constants, vacancy formation energies, and vacancy migrations.
- run from parent folder (e.g. do `julia ./scripts/elastic_constants.jl` for elastic constant calculation(s))
- ensure that required potential files are present in Potentials folder
- will place results into Outputs folder - either edit scripts or be careful of overwriting!

## FigureGeneration.ipynb
Notebook for reproducing figures from paper. Can be run using only given data in outputs folder.

## Outputs 
Results from paper contained in CSV files, used in FigureGeneration.ipynb.

## Figures
Figures from paper, can be reproduced with FigureGeneration.ipynb.