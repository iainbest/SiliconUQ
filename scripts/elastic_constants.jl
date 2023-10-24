### Set path to DFT data xyz file
datapath = "./data/gp_iter6_sparse9k.xml.xyz"

### Set QoI ; in this case, hard code elastic constants
quantity_function_string = "elasticconstants"

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

### set k_list, size of training data that potentials have been trained on
# k_list = [25,50,75,100,125,175,200,225,250,275,300,325]
k_list = [300]

### set desired coverage for conformal predictions
desired_coverage = 0.95

if Bpair
    Bpair_ = 3
else
    Bpair_ = "NONE"
end

### Since using python, slight change in how we pass potentials around
uses_python = true

### Set degree of site potential list
# deg_site_list = collect(3:25)
deg_site_list = [10]

### read in isolated atom energy to correct energies in larger configs later
isolated_atom = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", 
                                        virial_key="dft_virial", include=["isolated_atom";], verbose=false);
### and save as variable
isolated_atom_energy = isolated_atom[1].D["E"][1];


for k in k_list
    ### Set path to info dict (with calib,pred and train indices)
    infopath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/")

    ### Set path to relevant potential(s), not including deg site
    potpath = string("./potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/",string(species),"_degpair_$(Bpair_)_degsite_")               

    ### Set path to output results to
    outpath = "./outputs/elastic_constants"

    ### load in data used
    data = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial",
        include=incl);

    ### Set potentials to consider
    dbname_list = string.(potpath,deg_site_list)

    ### conformal - need to get train, calib indices
    info = DataFrame(CSV.File(infopath*"info"))

    ### get train, calib, test indices
    train_idx=info.actual_train_idx[findall(!ismissing,info.actual_train_idx)];
    calib_idx=info.calib_idx[findall(!ismissing,info.calib_idx)];
    test_idx=info.test_idx[findall(!ismissing,info.test_idx)];

    ### get calib and test data from above
    calib_data = data[calib_idx];
    test_data = data[test_idx];

    optim_strat = :particleswarm
    ### or read from outputs already done, e.g. read first row alpha and beta values, turn into matrix (be sure to access correct values this way)
    # optim_strat = Matrix(DataFrame(CSV.File("./outputs/bulk_modulus/dia$(k).csv"))[[1],[:α,:β]])

    outname = string(outpath,"/$(join(incl,"_"))$(k).csv")

    ### do elastic constants prediction with uncertainty for potentials above (which are all trained on same training data)
    ### this loop for a single potential size would reproduce a similar plot to the elastic constants in the paper
    basis_size_QoI_comparison_conformal_cov_scaling(dbname_list,outname,quantity_function_string,
                                                            isolated_atom_energy,species,ARD,calib_data,test_data,desired_coverage,false,
                                                            nothing,uses_python,optim_strat)
end