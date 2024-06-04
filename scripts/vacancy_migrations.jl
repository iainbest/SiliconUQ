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

### Set QoI ; in this case, hard code vacancy migration
quantity_function_string = "vacancymigration"

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
incl = ["dia","vacancy","divacancy"]

### if we only kept k configs... else comment out/ignore
k = 500

### set desired coverage for conformal predictions
desired_coverage = 0.95

if Bpair
    Bpair_ = 3
else
    Bpair_ = "NONE"
end

### Since using python, slight change in how we pass potentials around
uses_python = true

### Since only one potential, set to float not list
degsite = 20

### Set path to info dict (with calib,pred and train indices)
infopath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/")

### Set path to relevant potential(s), not including deg site
potpath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/",string(species),"_degpair_$(Bpair_)_degsite_")               

outpath = "./outputs/vacancy_migration"

### read in isolated atom energy to correct energies in larger configs later
isolated_atom = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", 
                                        virial_key="dft_virial", include=["isolated_atom";], verbose=false);
### and save as variable
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

outname = string(outpath,"/$(join(incl,"_"))$(k).csv")


### initial write of output csv dataframe
headings = ["dbname_list", "α", "β", "QoI_list", "evidence", "q̂_list"]
df = DataFrame([[] for i in headings],headings)
### initial write for headings
CSV.write(outname,df)

### set dbname depending on included configurations for training and degsite
dbname = "./potentials/correlation_order_$(corr_order)_EFV/$(join(incl,"_"))$(k)/$(string(species))_degpair_$(Bpair_)_degsite_$(degsite)"

### get pre-generated database
dB = LsqDB(dbname)

### get basis B so can read in inside QoIEvaluators.vacancymigration
B = basis(dB)

### for specific potential, get QoI output as well as hyperparameters, evidence and q̂ values
QoI_list,alpha,beta,evidence,q̂ = ACE_fs.ACE_QoI_DistributionConformalCovScaling(dB,isolated_atom_energy,
    quantity_function_string,species,ARD,calib_data,test_data,desired_coverage,false,nothing,uses_python,:particleswarm)

### write results out to CSV
CSV.write(outname,DataFrame([[dbname],[alpha],[beta],[QoI_list],[evidence],[q̂]],:auto),append=true)