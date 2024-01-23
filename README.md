# SiliconUQ

WIP notes

## Environment & Running the code

Manifest and Project toml files included for reproducible environment. 
To run scripts etc., need to 
- use Julia 1.8.3
- activate this same Julia environment with e.g. `using Pkg; Pkg.activate(".")` in this folder
- also required is to link this Julia to a Python environment with packages numpy, ASE, matscipy, Julia and pyjulip
- '
- Until instructions for linking julia to python update, can check bulk modulus script works as intended:
   - `julia ./scripts/B_increasing_basis.jl` should run and overwrite a simpler version of `./outputs/dia300.csv` (a single potential)

## Data 
Silicon 2018 GAP database from: 

A. P. Bart ́ok, J. Kermode, N. Bernstein, and G. Cs ́anyi, “Machine Learning a General-Purpose Interatomic Potential for Silicon,” Physical Review X 8, 41048 (2018), arXiv:1805.01568

https://github.com/libAtoms/silicon-testing-framework/blob/master/models/GAP/gp_iter6_sparse9k.xml.xyz

## Code 
- Julia and Python functions, classes etc. 
- Organised into code to perform QoI evaluations, and misc functions for building / optimising ACE potentials and analysis

## Potentials
- folder for holding generated potentials
- this is a mostly empty folder with one example potential built within using the `generate_simple_ACE_potentials.jl` script

## Scripts 
- example scripts to generate outputs for bulk modulus, elastic constants, vacancy formation energies, and vacancy migrations.
- run from parent folder (e.g. do `julia ./scripts/script.jl`)
- will place results into outputs folder - care for overwriting

## FigureGeneration.ipynb
- Notebook for reproducing figures from paper. Can be run using only given data in outputs folder.

## Outputs 
- results from paper as CSVs, used in FigureGeneration.ipynb.

## Figures
- Figures from paper, can be reproduced with FigureGeneration.ipynb.