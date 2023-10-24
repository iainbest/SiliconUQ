# SiliconUQ

WIP notes

Manifest and Project toml files included for reproducible environment. 
To run scripts etc., need to activate this same Julia environment. Also required is to link this Julia to a Python environment with numpy, ASE, matscipy, Julia and pyjulip 

Data - Silicon 2018 GAP database

Code - Julia and Python functions, classes etc. 

Scripts - example scripts to generate outputs for bulk modulus, elastic constants, vacancy formation energies, and vacancy migrations.
        - run from parent folder (e.g. do `julia ./scripts/script.jl`)
        - will place results into outputs folder - care for overwriting

Outputs - selected results from paper, used in FigureGeneration.ipynb.

FigureGeneration.ipynb - Jupyter notebook to reproduce selected figures from paper. 