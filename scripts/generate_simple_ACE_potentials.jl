### Check/print number of Threads
@show Threads.nthreads()

### Set observation type
observation_type = "EFV"
### Set path to data for training
datapath = "../data/gp_iter6_sparse9k.xml.xyz"
### Set correlation order
corr_order = 3
### Set species
species = :Si
### set nearest neighbour guess
r0 = rnn(species)
### Set if including pair basis part
Bpair = true
### Set if outputting potential as well
output_potential = false

### grab all si dataset data (or a selection of it...?)
incl = ["dia"]

### number of configs to keep in random 'split' (e.g. keep 100 random configs out of total)
k = 300

### Set output location
output_location = "../potentials/correlation_order_$(corr_order)_$(observation_type)/$(join(incl,"_"))$(k)/"

### Define some degree site list to generate potentials for, e.g.                                          
# deg_site_list = collect(4:24);
### for single potential:
deg_site_list = [3]

if Bpair
    Bpair_ = 3
else
    Bpair_ = "NONE"
end

### read in isolated atom energy to correct energies in larger configs later...and save as variable
isolated_atom_energy = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial", include=["isolated_atom";], verbose=false)[1].D["E"][1];

data = IPFitting.Data.read_xyz(datapath, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial",include=incl);

### for conformal procedure: split configs into train/calibrate/test.#####################################################

### collect atomic data indices
data_idx = collect(1:length(data));
### set rng for reproducibility
rng = MersenneTwister(123);
### keep 80% for training, 20% for testing
train_idx,test_idx = traintestsplit(data_idx,0.8,rng);
### for conformal approaches, split train set into true train and calibration set
### in this case, 90% of previous split kept for training (~72% of total data)
train_idx,calib_idx = traintestsplit(train_idx,0.9,rng);

### get k configs out of train_idx, train only on them (where k is [1:k] in below)
### we will eventually calibrate on the same calib_idx data regardless of k
train_data = data[train_idx[1:k]]
dataset_cfgs = train_data

### write conformal splits to dataframe
df = DataFrame()

### add missing values so that columns are same length

df.data_idx = data_idx
df.full_train_idx_split = [train_idx;[missing for _ in 1:(length(data_idx)-length(train_idx))]]
df.actual_train_idx = [train_idx[1:k];[missing for _ in 1:(length(data_idx)-length(train_idx[1:k]))]]
df.calib_idx = [calib_idx;[missing for _ in 1:(length(data_idx)-length(calib_idx))]]
df.test_idx = [test_idx;[missing for _ in 1:(length(data_idx)-length(test_idx))]]
df[!,"k"] .= k

CSV.write(string(output_location,"info"),df)

####################################################################################################################

### for each entry in deg_site_list, build an ACE potential
for deg_site in deg_site_list

    println("On degree ",deg_site)

    ### If potential exists already, these files will exist
    kron_loc = string(output_location,string(species),
                        "_degpair_$(Bpair_)_degsite_$(deg_site)_kron.h5")

    info_loc = string(output_location,string(species),
                        "_degpair_$(Bpair_)_degsite_$(deg_site)_info.json")

    ### if files exist: do nothing
    if isfile(kron_loc) && isfile(info_loc)
        println("Database already exists: continuing...")
        continue
    ### else build database!
    else
        ### actually build potential
        build_ACE_potential(species,dataset_cfgs,isolated_atom_energy,corr_order,deg_site,output_location,Bpair,output_potential)
    end

end
