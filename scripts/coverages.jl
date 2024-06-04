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

data = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", 
                    virial_key="dft_virial", include=["dia"], 
                    verbose=false);


### set paths to (conformal) info file and to specific potential
infopath = "./potentials/correlation_order_3_EFV/dia300/"
potpath = "./potentials/correlation_order_3_EFV/dia300/Si_degpair_3_degsite_3"

### having already done hyperparam optim, provide values for alpha and beta
optim_strat = [2.3967995185539165e-7 85.08520706444692;];

### 
dB_emp = LsqDB(potpath);
B_emp = basis(dB_emp);

weights_ = Dict("default" => Dict( "E" => 30.0, "F" => 1.0, "V" => 1.0 ),
                    "liq" => Dict( "E" => 10.0, "F" => 0.66, "V" => 0.25 ),
                    "amorph" => Dict( "E" => 3.0, "F" => 0.5, "V" => 0.1 ),
                    "sp" => Dict( "E" => 3.0, "F" => 0.5, "V" => 0.1 ),
                    "bc8" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "vacancy" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "divacancy" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "interstitial" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ))


### if conformal - need to get train, calib atomic indices and convert them to EFV indices
### get info file:
info = DataFrame(CSV.File(infopath*"info"))

### get ATOMIC train, calib, test indices
train_idx=info.actual_train_idx[findall(!ismissing,info.actual_train_idx)];
calib_idx=info.calib_idx[findall(!ismissing,info.calib_idx)];
test_idx=info.test_idx[findall(!ismissing,info.test_idx)];

### get calib and test data from above
calib_data = data[calib_idx];
test_data = data[test_idx];

### set desired coverage
desired_coverage = 0.95

### setup conformal problem
ζ,n,n_val,l = conformalsetup(desired_coverage,calib_idx,test_idx)


### assemble full test set = calibration + test sets, get EFV data in vector

full_test_idx = vcat(calib_idx,test_idx);
full_test_data = data[full_test_idx];
full_test_configs = getfield.(full_test_data,:at);

### get database of calibration basis vs configs
dB_ = LsqDB("",B_emp,calib_data)
### get observations of calibration set
a = IPFitting.Lsq.collect_observations(dB_,IPFitting.Lsq._fix_weights!(nothing),nothing)
calib_data_efv = a[1]

### get database of prediction/test basis vs configs
dB_ = LsqDB("",B_emp,test_data)
### get observations of test set
a = IPFitting.Lsq.collect_observations(dB_,IPFitting.Lsq._fix_weights!(nothing),nothing)
test_data_efv = a[1];

### combine into full vector
full_test_efv = vcat(calib_data_efv,test_data_efv);


### make linear system
species = :Si
n_samples = 100

Φ,y_vector = IPFitting.Lsq.get_lsq_system(dB_emp,Vref=OneBody(species => isolated_atom_energy),weights=weights_);

alpha,beta = optim_strat

### calculate posterior distribution
mN,SN,SN_inv = posterior(Φ,y_vector,alpha,beta);

### generate samples from posterior - these are weights / coefficients of potential
SN_inv = Matrix(Hermitian(SN_inv))
coeff_generated = rand(MvNormalCanon(SN_inv*mN, SN_inv), n_samples);


### assemble matrix of predictions
test_QoI_list = Vector{Vector{Float64}}(undef,n_samples)

for i in 1:n_samples
    
    ### get vector of weights from posterior distribution
    coeffs = coeff_generated[:,i]

    ### build ACE potential from coefficient and basis
    V = build_ACE_from_coefficients(coeffs,B_emp,species,isolated_atom_energy)
    
    ### evaluate all efv to save doing multiple loops
    test_values = QoIEvaluators.efv(V,species,full_test_configs)
    
    ### append output of QoI calc for specific potential to list
    test_QoI_list[i] = test_values
    
end

test_QoI_list = vecvec_to_matrix(test_QoI_list)

### set number of repeats to perform
R = 1000

### for each R, shuffle calibration and test set, and calculate empirical coverage
coverages = empiricalcoverages(R,full_test_efv,test_QoI_list,length(calib_idx)/length(test_idx),ζ)

### save to file (commented to avoid overwriting)
# CSV.write("./outputs/coverages/dia300_degsite_3.csv",DataFrame([[ζ_],[n_],[n_val_],[l_],[coverages]],:auto),append=true)