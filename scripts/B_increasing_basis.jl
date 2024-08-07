using ACE1pack, ACE1, IPFitting
using Statistics
using Random
using LinearAlgebra
using Distributions
using Plots
using StatsPlots
using CSV, DataFrames

using IB_BayesianLinearRegression
using UQPotentials
using ConformalPredictions

### Use code snippets
include("../code/ACE_fs.jl")
using .ACE_fs

include("../code/QoIEvaluators.jl")
using .QoIEvaluators

### Set path to DFT data xyz file
datapath = "./data/gp_iter6_sparse9k.xml.xyz"

### Set QoI ; in this case, hard code bulk modulus
quantity_function_string = "bulkmodulus"

### Set observation type
observation_type = "EFV"

### Set species
species = :Si

ARD = false

### Set correlation order
corr_order = 3

### Set if including pair basis part
Bpair = true

### Set data to include
incl = ["dia"]

### training on k configs
k = 300

### set desired coverage for conformal predictions
desired_coverage = 0.95

if Bpair
    Bpair_ = 3
else
    Bpair_ = "NONE"
end

### Set degree of site potential list
# deg_site_list = collect(3:24)
deg_site_list = [3]

### Set path to info dict (with calib,pred and train indices)
infopath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/")

### Set path to relevant potential(s)
potpath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/",string(species),"_degpair_$(Bpair_)_degsite_")               

### Set path to output results to
outpath = "./outputs/bulk_modulus"

### read in isolated atom energy to correct energies in larger configs later...
isolated_atom = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", 
                                            force_key="dft_force", virial_key="dft_virial", include=["isolated_atom";], verbose=false);
### ...and save as variable
isolated_atom_energy = isolated_atom[1].D["E"][1];

### load in data used
data = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial",
    include=incl);


### conformal - need to get train, calib indices
### get info file:
info = DataFrame(CSV.File(infopath*"info"))

### get train, calib, test indices
train_idx=info.actual_train_idx[findall(!ismissing,info.actual_train_idx)];
calib_idx=info.calib_idx[findall(!ismissing,info.calib_idx)];
test_idx=info.test_idx[findall(!ismissing,info.test_idx)];

### get calib and test data from above
calib_data = data[calib_idx];
test_data = data[test_idx];

### Set potentials to consider
dbname_list = string.(potpath,deg_site_list)
outname = string(outpath,"/$(join(incl,"_"))$(k).csv")

### do bulk modulus prediction with uncertainty to potentials above (which are all trained on same training data)
### this is a slice of the heatmap presented in the paper
basis_size_QoI_comparison_conformal_cov_scaling(dbname_list,outname,quantity_function_string,
                            isolated_atom_energy,species,ARD,calib_data,test_data,desired_coverage,false,nothing,false,:particleswarm)
