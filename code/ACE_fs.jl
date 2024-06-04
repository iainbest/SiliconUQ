module ACE_fs

using ASE, ACE1, JuLIP, IPFitting, ACE1x, ACEfit
using LinearAlgebra, Distributions, Optim, LineSearches, LatinHypercubeSampling, Polynomials
using IB_BayesianLinearRegression, UQPotentials, ConformalPredictions
using DataFrames, CSV, Plots, PyCall
using HDF5: h5open, read
using ProgressMeter

include("./QoIEvaluators.jl")
using .QoIEvaluators

### python imports and definitions
ase = pyimport("ase")
pyimport("ase.optimize")
pyimport("ase.mep")              ### changed from neb -> mep
pyjulip = pyimport("pyjulip")

QuasiNewton = ase.optimize.QuasiNewton
NEB = ase.mep.NEB              ### changed from neb -> mep
NEBOptimizer = ase.mep.neb.NEBOptimizer              ### changed from neb -> mep.neb
NEBTools = ase.mep.neb.NEBTools              ### changed from neb -> mep

export build_ACE_potential,basis_lengths,basis_size_bulk_modulus_comparison
export load_kron
export basis_size_QoI_comparison
export unpack_VacancyMigration_output
export build_ACE_from_coefficients, ACE_QoI_Distribution

export log_marginal_likelihood_overdetermined!
export basis_size_QoI_comparison_mixed
export basis_size_QoI_comparison_conformal_cov_scaling



"""
    build_ACE_potential(species,cfgs,isolated_atom_energy,N,deg_site,output_location,include_Bpair=true,output_potential=false)

Build an ACE potential given a set of atomic configurations, maximum polynomial degree, and whether
to include pair terms.

### Arguments

- `species::Symbol`: atomic species, e.g. `:Si`
- `cfgs::Vector{Dat}`: atomic configurations. CHECK TYPE (might be Vector{Atoms{Float64}} sometimes?? maybe?)
- `isolated_atom_energy`: energy of onebody isolated atom
- `N::Int64`: correlation order for ACE portion of the basis. Correlation order = body order - 1
- `deg_site::Int64`: degree of site potential polynomial
- `output_location::String`: folder / location to write potential and other files to, e.g. `"../Generated_ACE_Potentials"`. 
- `include_Bpair::Bool`: true or false, choice to include pair potential in the basis. Default true
- `output_potential::Bool`: true or false, choice to output potential or not. Default false

Weights on energies, forces, virials has not been set - these are applied later. Note that this is not the most general or correct way to 
build an ACE potential - see ACEpotentials.jl docs. 

"""
function build_ACE_potential(species::Symbol,cfgs,onebody_energy,N,deg_site::Int64,output_location::String,include_Bpair=true,output_potential=false)
    deg_pair = 3
    r0 = rnn(species)
    
    ### many body part
    Bsite = rpi_basis(species = species,  
                        N = N,                     # correlation order = body-order - 1
                        maxdeg = deg_site,         # polynomial degree
                        r0 = r0,                   # estimate for nearest neighbour dist
                        rin = 0.8*r0, rcut = 2*r0, # cutoffs
                        pin = 2)

    ### cubic pair potential part
    Bpair = pair_basis(species = species, r0 = r0, maxdeg = deg_pair, 
                        rcut = 3 * r0, rin = 0.0, # cutoffs 
                        pin = 0 )                 # pin = 0 means no inner cutoff
    
                        ### combine pair and many body
    B_pair_and_site = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);


    if include_Bpair == true
        ### choose superbasis including pair energies
        B = B_pair_and_site
        ### set database file name
        dbname = string(output_location,"/",string(species),"_degpair_$(deg_pair)_degsite_$(deg_site)")
        # dbname = "../Generated_ACE_Potentials/Si_degpair_$(deg_pair)_degsite_$(deg_site)"
    else
        ### choose rpibasis with only site energies
        B = Bsite
        ### set database file name
        dbname = string(output_location,"/",string(species),"_degpair_NONE_degsite_$(deg_site)")
        # dbname = "../Generated_ACE_Potentials/Si_degpair_NONE_degsite_$(deg_site)"
    end


    ### build database and save to location dbname
    dB = LsqDB(dbname, B, cfgs);

    if output_potential == true
    
        ### set weights on energies, forces and virials (stresses)
        weights = Dict("default" => Dict("E" => 1.0, "F" => 1.0 , "V" => 1.0 ))

        if onebody_energy !== nothing
            ### Add in reference onebody potential for isolated Si atom
            Vref = OneBody(Dict(string(species) => onebody_energy))
        else
            Vref=nothing
        end

        ### perform the least squares fit to build potential and info data
        IP, lsqinfo = lsqfit(dB; weights = weights, Vref = Vref, asmerrs = true);

        ### set potential name and save it
        potname = "$(dbname)_potential.json"
        save_dict(potname, Dict("IP" => write_dict(IP), "info" => lsqinfo))
        
    else  
    end
end

"""
    build_ACE_from_coefficients(coeffs,basis,species,isolated_atom_energy)

Builds an ACE potential from a basis set and a vector of coefficients.

### Arguments 

- `coeffs::Vector{Float64}`: vector of potential coefficients
- `B::JuLIP.MLIPs.IPSuperBasis{JuLIP.MLIPs.IPBasis}`: potential basis
- `species::Symbol`: the chemical species of the atoms (only single species so far)
- `isolated_atom_energy::Float64`: energy of isolated atom, used for reference potential

"""
function build_ACE_from_coefficients(coeffs,B,species,isolated_atom_energy)

    ### reference onebody potential
    Vref = OneBody(Dict(species => isolated_atom_energy))

    ### build ACE potential from coefficents, basis, and reference energy
    V = JuLIP.MLIPs.SumIP(Vref,JuLIP.MLIPs.combine(B,coeffs));

    return V
end


function basis_lengths(dbname_list::Vector{String})
    basis_lengths = fill(-1.0,length(dbname_list))
    for (count,dbname) in enumerate(dbname_list)
        dB = LsqDB(dbname)
        B = dB.basis
        basis_lengths[count] = length(B)
    end
    return basis_lengths
end
function data_lengths(dbname_list::Vector{String})
    data_lengths = fill(-1.0,length(dbname_list))
    for (count,dbname) in enumerate(dbname_list)
        dB = LsqDB(dbname)
        Φ,y = IPFitting.Lsq.get_lsq_system(dB,Vref=nothing,weights=nothing)
        data_lengths[count] = length(y)
    end
    return data_lengths
end

function get_qhat(calib_data,pred_data,coeff_generated,B,isolated_atom_energy,weights_,desired_coverage,species,n_samples=50)

    ### get database of calibration basis vs configs
    dB_ = LsqDB("",B,calib_data)

    ### get true observations for calibration set
    a = IPFitting.Lsq.collect_observations(dB_,IPFitting.Lsq._fix_weights!(nothing),nothing)
    calib_data_efv = a[1]

    ### get database of prediction/test basis vs configs
    dB_ = LsqDB("",B,pred_data)

    ### get true observations for prediction set
    a = IPFitting.Lsq.collect_observations(dB_,IPFitting.Lsq._fix_weights!(nothing),nothing)
    pred_data_efv = a[1]

    calib_idx = collect(1:length(calib_data_efv));
    pred_idx = collect(1:length(pred_data_efv));

    ### setup conformal problem
    ### in this case, calib_idx and pred_idx will be labelled as E,F,V observations
    ζ,n,n_val,l = conformalsetup(desired_coverage,calib_idx,pred_idx)

    ### initialize list
    calib_QoI_list = Vector{Vector{Float64}}(undef,n_samples)

    ### from coeff_generated,B calculate E,F,V on calibration set
    for i in 1:n_samples

        ### get vector of weights from posterior distribution
        coeffs = coeff_generated[:,i]

        ### build ACE potential from coefficient and basis
        V = build_ACE_from_coefficients(coeffs,B,species,isolated_atom_energy)

        ### evaluate efv
        test_values = QoIEvaluators.efv(V,species,calib_data)

        ### append output of QoI calc for specific potential to list
        calib_QoI_list[i] = test_values[calib_idx]

    end

    ### convert vecvec to matrix
    calib_QoI_list = vecvec_to_matrix(calib_QoI_list)

    ### calculate mean(s) and stdev(s) of calibration set
    calib_mean_list = [mean(calib_QoI_list[:,i]) for i in 1:size(calib_QoI_list)[2]];
    calib_std_list = [std(calib_QoI_list[:,i]) for i in 1:size(calib_QoI_list)[2]];

    ### calculate scores on calibration set
    scores = calibrationscores(calib_data_efv,calib_mean_list,calib_std_list)

    ### calculate quantile for q̂
    q_val = qval(n,ζ)

    ### calculate q̂ 
    q̂ = qhat(scores,q_val);

    return q̂
end

### main workhorse of UQ in code: given dB defining a potential, quantity_function_string defining a QoI, and some other arguments, 
### perform hyperparameter optimisation, evidence calc, q̂ calc, and calc QoI for each ensemble member.
function ACE_QoI_DistributionConformalCovScaling(dB,isolated_atom_energy,quantity_function_string,species,ARD,
    calib_data,pred_data,desired_coverage,cov_scaling::Bool,fraction=nothing,uses_python=false,
    optim_strat=:particleswarm)
    
    ### modify quantity function from string to function
    quantity_function = getfield(QoIEvaluators,Symbol(quantity_function_string))

    ### get reduced function which calcs QoI from relaxed configurations
    quantity_function_reduced = getfield(QoIEvaluators,Symbol(quantity_function_string*"reduced"))

    ### Grab type of output quantity for array initialisation
    output_type = Base.return_types(quantity_function_reduced)[1]
    
    ### Get basis B from database dB
    B = basis(dB)

    ### set weights (from pace paper supp details, added divacancy values to match vacancy)
    weights_ = Dict("default" => Dict( "E" => 30.0, "F" => 1.0, "V" => 1.0 ),
                    "liq" => Dict( "E" => 10.0, "F" => 0.66, "V" => 0.25 ),
                    "amorph" => Dict( "E" => 3.0, "F" => 0.5, "V" => 0.1 ),
                    "sp" => Dict( "E" => 3.0, "F" => 0.5, "V" => 0.1 ),
                    "bc8" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "vacancy" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "divacancy" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ),
                    "interstitial" => Dict( "E" => 50.0, "F" => 0.5, "V" => 0.1 ))

    ### get design matrix, observation vector
    Φ, y_vector = IPFitting.Lsq.get_lsq_system(dB,Vref=OneBody(species => isolated_atom_energy),weights=weights_)

    ### different choices of optimisation of hyperparameters here: e.g. mackay updates, l-bfgs, particle swarm, read in
    ### each takes in Φ,y, some specific args...outputs optimised alpha, beta
    if optim_strat == :mackayupdates
        ### start points for hyperparam optimisation, for now keep consistent between runs & set to one
        if ARD
            initial_α = ones(length(B))
        else
            initial_α = 1.0
        end
        initial_β = 1.0

        ### perform evidence approximation for hyperparameters
        alpha_list,beta_list = evidence_approximation(Φ,y_vector,initial_α,initial_β)

        ### set values for hyperparameters according to evidence approx
        alpha = alpha_list[end]
        beta = beta_list[end];
    elseif optim_strat == :particleswarm
        ### if ARD then dims of LHC change
        if ARD
            n_dims = length(B) + 1
        else
            n_dims = 2
        end

        n_points = 50
        n_iter = 500
        XTX = Φ'*Φ

        ### initialize particles - use LHC points
        lhc, _ = LHCoptim(n_points,n_dims,10)
        ### rescale LHC between ~zero and upper value (which scales? with system size), in log units
        lhc_s = scaleLHC(lhc,repeat([(-20,20)],n_dims));

        ### initialize population at LHC sites, 0 velocity
        population = Vector{Particle}(undef,n_points)
        for i in 1:n_points
            population[i] = Particle(lhc_s[i,:],0.0,zeros(n_dims),lhc_s[i,:],0.0,[lhc_s[i,:]])  
        end

        new_population,x_best = particle_swarm_optimization(negative_lml, population, n_iter, Φ, y_vector, XTX, w = 0.9, c1 = 1, c2 = 1)
        x_best = exp.(x_best);

        if ARD
            alpha = x_best[1:end-1]
            beta = x_best[end]
        else
            alpha = x_best[1]
            beta = x_best[2]
        end
    elseif optim_strat == :lbfgs
        if ARD
            ### ??? TODO ARD
        else
            bf_mN,bf_α,bf_β,bf_evidence,bf_trace,all_traces,initial_hypers = bayesian_fit_restarts(y_vector,Φ,50)
        end
        alpha = bf_α
        beta = bf_β

    elseif typeof(optim_strat) == Vector{Float64}
        if ARD
            @assert length(optim_strat) == length(B) + 1
            alpha = optim_strat[1:end-1]
            beta = optim_strat[end]
        else
            @assert length(optim_strat) == 2
            alpha = optim_strat[1]
            beta = optim_strat[end]
        end
    else
        ### ??? TODO
    end

    ### calculate posterior distribution
    mN,SN,SN_inv = posterior(Φ,y_vector,alpha,beta);
    
    ### evaluate evidence / log marginal likelihood for comparison across different potentials and append to list
    evidence = log_marginal_likelihood_overdetermined!(0.0,Φ,y_vector,alpha,beta,Φ'*Φ)

    ### set number of samples
    n_samples = 100
    
    ### set up container for QoI evaluations
    QoI_list = Vector{output_type}(undef,n_samples)
    
    ### generate samples from posterior - these are weights / coefficients of potential
    SN_inv = Matrix(Hermitian(SN_inv))
    coeff_generated = rand(MvNormalCanon(SN_inv*mN, SN_inv), n_samples)  # equivalent to rand(MvNormal(mN,SN), n_samples)
    
    ### get qhat
    q̂ = get_qhat(calib_data,pred_data,coeff_generated,B,isolated_atom_energy,weights_,desired_coverage,species,n_samples)
    
    if cov_scaling
        ### scale covariance matrix (inverse) by q̂^2 (by division)
        SN_inv_ = Matrix(Hermitian(SN_inv / (q̂^2)))
        coeff_generated_ = rand(MvNormalCanon(SN_inv_*mN, SN_inv_), n_samples);
    else
        coeff_generated_ = coeff_generated
    end
    
    ### TODO ARD here 

    if uses_python
        for i in 1:n_samples

            ### get vector of weights from posterior distribution
            coeffs = coeff_generated[:,i]
    
            ### append output of QoI calc for specific potential to list
            QoI_list[i] = quantity_function(coeffs,B,species,isolated_atom_energy)
        end
    else
        for i in 1:n_samples

            ### get vector of weights from posterior distribution
            coeffs = coeff_generated[:,i]
    
            ### build ACE potential from coefficient and basis
            V = build_ACE_from_coefficients(coeffs,B,species,isolated_atom_energy)
    
            ### append output of QoI calc for specific potential to list
            QoI_list[i] = quantity_function(V,species)#[1]
    
        end
    end
    
    return QoI_list,alpha,beta,evidence,q̂
end

### do conformal QoI calcs for different potetnials defined by dbname_list, set up and run ACE_QoI_DistributionConformalCovScaling function,
### and write results to DataFrame outname
function basis_size_QoI_comparison_conformal_cov_scaling(dbname_list::Vector{String},outname::String,
    quantity_function_string::String,isolated_atom_energy::Float64,species::Symbol,
    ARD::Bool,calib_data::Vector{Dat},pred_data::Vector{Dat},desired_coverage,
    cov_scaling,fraction=nothing,uses_python=false,optim_strat=:particleswarm)

    ### make empty dataframe with correct headings
    headings = ["dbname_list", "α", "β", "QoI_list", "evidence", "q̂_list"]
    df = DataFrame([[] for i in headings],headings)
    CSV.write(outname,df)
 
    ### for each different potential / basis set
    for (count,dbname) in enumerate(dbname_list)

        if typeof(optim_strat) != Symbol
            optim_strat_ = optim_strat[count,:]
        else
            optim_strat_=optim_strat
        end
 
        println("On potential $(count) / $(length(dbname_list)).")

        ### get pre-generated Least-squares fit database
        dB = LsqDB(dbname)

        ### get basis from database
        B = dB.basis

        ### for specific potential, get QoI output distribution with scaled cov matrix
        QoI_list,alpha,beta,evidence,q̂ = ACE_QoI_DistributionConformalCovScaling(dB,isolated_atom_energy,
            quantity_function_string,species,ARD,calib_data,pred_data,desired_coverage,cov_scaling,fraction,uses_python,optim_strat_)
    
        ### append to dataframe
        CSV.write(outname,DataFrame([[dbname],[alpha],[beta],[QoI_list],[evidence],[q̂]],:auto),append=true)
 
    end
    return
 
end

############### Following is code that has been adapted or borrowed from various sources, see links in relevant sections ##############

### from ipfitting: https://github.com/ACEsuit/IPFitting.jl/blob/master/src/lsq_db.jl ###############################################
const KRONFILE = "_kron.h5"
kronfile(dbpath::AbstractString) = dbpath * KRONFILE
"load a single matrix from HDF5"
_loadmath5(fname) =
   h5open(fname, "r") do fid
      read(fid["A"])
   end
load_kron(dbpath::String; mmap=false) = _loadmath5(kronfile(dbpath))
load_kron(db::LsqDB; mmap=false) = load_kron(dbpath(db); mmap=mmap)
#######################################################################################################################################

######## FROM ACEfit.jl https://github.com/ACEsuit/ACEfit.jl/blob/main/src/bayesianlinear.jl ############################################
### changed from variances to precision!
function log_marginal_likelihood_overdetermined!(
    lml::AbstractFloat,
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    α::AbstractFloat,
    β::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    α_vec = α * ones(size(X,2))
    lml = log_marginal_likelihood_overdetermined!(lml, X, y, α_vec, β, XTX)
    return lml
end

function log_marginal_likelihood_overdetermined!(
    lml::AbstractFloat,
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    α::Vector{<:AbstractFloat},
    β::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    N = size(X,1)
    M = size(X,2)
    Σ_c = Array{Float64}(undef,M,M)
    
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX,1), Σ_c, stride(Σ_c,1))
    BLAS.scal!(length(Σ_c), β, Σ_c, stride(Σ_c,1))
    for i in 1:M; Σ_c[i,i] += α[i]; end
    
    C = cholesky!(Symmetric(Σ_c))
    Σ_c = C \ I(M)
    μ_c = β*(C \ (X'*y))
    
    lml = - 0.5*logdet(C) - 0.5*sum(log.(1 ./α)) - 0.5*N*log(1/β) - 0.5*N*log(2*π)
    lml -= 0.5*β*y'*(y-X*μ_c)
    
    return lml
end

#########################################################################################################################################
#### For L-BFGS routines optim_strat. also adapted from ACEfit.jl

function solve(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat,
)
    return solve(y, X, var_c*ones(size(X,2)), var_e)
end

function solve(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
)
    M = size(X,2)
    XTX = X'*X
    Σ_c = Array{Float64}(undef,M,M)
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX,1), Σ_c, stride(Σ_c,1))
    BLAS.scal!(length(Σ_c), 1.0/var_e, Σ_c, stride(Σ_c,1))
    for i in 1:M; Σ_c[i,i] += 1.0/var_c[i]; end
    C = cholesky!(Symmetric(Σ_c))
    return 1.0/var_e*(C \ (X'*y))
end

function log_marginal_likelihood_overdetermined!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    var_c_vec = var_c*ones(size(X,2))
    grad_vec = zeros(size(X,2)+1)
    lml = log_marginal_likelihood_overdetermined!(lml, grad_vec, X, y, var_c_vec, var_e, XTX)
    grad[1] = sum(grad_vec[1:end-1])
    grad[2] = grad_vec[end]
    return lml
end

function log_marginal_likelihood_overdetermined!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    N = size(X,1)
    M = size(X,2)
    Σ_c = Array{Float64}(undef,M,M)
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX,1), Σ_c, stride(Σ_c,1))
    BLAS.scal!(length(Σ_c), 1.0/var_e, Σ_c, stride(Σ_c,1))
    for i in 1:M; Σ_c[i,i] += 1.0/var_c[i]; end

    C = try
        C = cholesky!(Symmetric(Σ_c))
        catch e
        if typeof(e) <: PosDefException
            return -Inf
        else
            rethrow()
        end
    end
    
    Σ_c = C \ I(M)
    μ_c = 1.0/var_e*(C \ (X'*y))
    lml = - 0.5*logdet(C) - 0.5*sum(log.(var_c)) - 0.5*N*log(var_e) - 0.5*N*log(2*π)
    lml -= 0.5/var_e*y'*(y-X*μ_c)
    grad[1:M] .= 0.5*(μ_c.^2 .+ diag(Σ_c) .- var_c)./var_c.^2 
    grad[M+1] = 0.5/var_e^2*(sum((y-X*μ_c).^2) + dot(XTX,Σ_c) - N*var_e)
    return lml
end

function log_marginal_likelihood_underdetermined!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat,
)
    var_c_vec = var_c*ones(size(X,2))
    grad_vec = zeros(size(X,2)+1)
    lml = log_marginal_likelihood_underdetermined!(lml, grad_vec, X, y, var_c_vec, var_e)
    grad[1] = sum(grad_vec[1:end-1])
    grad[2] = grad_vec[end]
    return lml
end

function log_marginal_likelihood_underdetermined!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
)
    N = size(X,1)
    M = size(X,2)
    Σ_y = X*Diagonal(var_c)*X'
    for i=1:N; Σ_y[i,i] += var_e; end
    C = cholesky!(Symmetric(Σ_y))
    invΣy_y = C \ y
    lml = -0.5*y'*invΣy_y - 0.5*logdet(C) - 0.5*N*log(2*π)
    grad[1:M] .= 0.5*(X'*invΣy_y).^2
    W = C \ X
    @views for i=1:M; grad[i] -= 0.5*dot(X[:,i], W[:,i]); end
    grad[M+1] = 0.5*dot(invΣy_y,invΣy_y) - 0.5*tr(C\I(N))
    return lml
end

function bayesian_fit(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat};
    variance_floor::AbstractFloat=1e-30,
    verbose::Bool=false,
    initial_hypers::Vector{<:AbstractFloat}=ones(2),
)
    if size(X,1) > size(X,2)
        XTX = X'*X  # advantageous to precompute for overdetermined case
    end

    function fg!(f, g, x)
        var_c = variance_floor + x[1]*x[1]
        var_e = variance_floor + x[2]*x[2]
        if size(X,1) >= size(X,2)
            f = log_marginal_likelihood_overdetermined!(f, g, X, y, var_c, var_e, XTX)
        else
            f = log_marginal_likelihood_underdetermined!(f, g, X, y, var_c, var_e)
        end
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2*x
        end
        return f
    end
    
    res = optimize(Optim.only_fg!(fg!),
                   initial_hypers,
                   Optim.LBFGS(linesearch = MoreThuente()),
                   Optim.Options(x_tol=1e-8, g_tol=0.0, show_trace=verbose,
                                extended_trace=true,store_trace=true))
    
    verbose && println(res)
    
    ### get trace positions (x_trace) out from extended_trace metadata
    metadata_ = getfield.(res.trace,:metadata)
    x_trace = [variance_floor .+ i["x"].*i["x"] for i in metadata_]
    

    lml = -Optim.minimum(res)
    var_c, var_e = Optim.minimizer(res)
    var_c = variance_floor + var_c*var_c
    var_e = variance_floor + var_e*var_e

    return solve(y, X, var_c, var_e), var_c, var_e, lml, x_trace
end

function bayesian_fit_restarts(y_vector,Φ,n_restarts=20)
    
    ### initialize values
    mN,inv_α,inv_β,evidence,trace = ones(size(Φ)[2]),1.0,1.0,-Inf,nothing
    bf_mN,bf_inv_α,bf_inv_β,bf_evidence,bf_trace = ones(size(Φ)[2]),1.0,1.0,-Inf,nothing
    
    ### make matrix of LHC points for initial hyper points, log precision units
    lhc, _ = LHCoptim(n_restarts,2,10)
    ### rescale LHC between ~zero and upper value
    lhc_s = scaleLHC(lhc,[(-20,20),(-20,20)]);
    
    ### container for all traces
    all_traces = []
    
    for n in 1:n_restarts
        
        ### need to catch positive definite errors - try bayesian fit first, initialized at random point
        try
            bf_mN,bf_inv_α,bf_inv_β,bf_evidence,bf_trace = bayesian_fit(y_vector,Φ,initial_hypers=1.0./(exp.(lhc_s[n,:])))
            catch e
            if typeof(e) <: PosDefException
                ### if matrix not positive definite, then do not update
                bf_mN,bf_inv_α,bf_inv_β,bf_evidence,bf_trace = ones(size(Φ)[2]),1.0,1.0,-Inf,nothing
            else
                ### rethrow different error
                rethrow()
            end
        end
        
        ### catch all traces done and push
        push!(all_traces,bf_trace)
        
        ### update values if better than previous
        if bf_evidence > evidence
            mN,inv_α,inv_β,evidence,trace = bf_mN,bf_inv_α,bf_inv_β,bf_evidence,bf_trace
        else
        end
    end
    
    return mN,1/inv_α,1/inv_β,evidence,trace,all_traces,lhc_s
end

#########################################################################################################################################
#### Particle swarm optimisation taken & adapted from Algorithms for Optimization (Kochenderfer & Wheeler)
#### lml functions from above & adapted to give grads

function log_marginal_likelihood_overdetermined(
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    grad = zeros(2)
    
    var_c_vec = var_c*ones(size(X,2))
    
    lml,grad_vec = log_marginal_likelihood_overdetermined(X, y, var_c_vec, var_e, XTX)
    
    grad[1] = sum(grad_vec[1:end-1])
    grad[2] = grad_vec[end]
    
    return lml,grad
end

function log_marginal_likelihood_overdetermined(
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
    XTX::Matrix{<:AbstractFloat},
)
    grad_vec = zeros(size(X,2)+1)
    N = size(X,1)
    M = size(X,2)
    Σ_c = Array{Float64}(undef,M,M)
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX,1), Σ_c, stride(Σ_c,1))
    BLAS.scal!(length(Σ_c), 1.0/var_e, Σ_c, stride(Σ_c,1))
    for i in 1:M; Σ_c[i,i] += 1.0/var_c[i]; end
    
    C = try
        C = cholesky!(Symmetric(Σ_c))
        catch e
        if typeof(e) <: PosDefException
            ### check this
            return -Inf,[0.0,0.0]
        else
            rethrow()
        end
    end
    
    Σ_c = C \ I(M)
    μ_c = 1.0/var_e*(C \ (X'*y))
    lml = - 0.5*logdet(C) - 0.5*sum(log.(var_c)) - 0.5*N*log(var_e) - 0.5*N*log(2*π)
    lml -= 0.5/var_e*y'*(y-X*μ_c)
    grad_vec[1:M] .= 0.5*(μ_c.^2 .+ diag(Σ_c) .- var_c)./var_c.^2 
    grad_vec[M+1] = 0.5/var_e^2*(sum((y-X*μ_c).^2) + dot(XTX,Σ_c) - N*var_e)
    
    return lml,grad_vec
end


mutable struct Particle
    x
    y
    v
    x_best
    y_best
    path
end


function particle_swarm_optimization(f, population, k_max, Φ, y_vector, XTX; w=1, c1=1, c2=1, export_paths=false)
    n = length(population[1].x)
    x_best, y_best = copy(population[1].x_best), Inf
    for P in population
        y = f(P.x,Φ,y_vector,XTX)
        if y < y_best; x_best[:], y_best = P.x, y; end
    end
    for k in 1 : k_max
        Threads.@threads for P in population
            r1, r2 = rand(n), rand(n)
            P.x += P.v
            P.v = w*P.v + c1*r1.*(P.x_best - P.x) + c2*r2.*(x_best - P.x)
            P.y = f(P.x,Φ,y_vector,XTX)
            if export_paths; push!(P.path,P.x); end
            if P.y < P.y_best; P.x_best[:] = P.x; end
        end

        ### update y_best, x_best here
        min_pos = argmin(getfield.(population,:y))
        
        if population[min_pos].y < y_best
            y_best = population[min_pos].y
            x_best = population[min_pos].x
        end
    end
    return population,x_best
end

### working with x in log units now!
function negative_lml(x,Φ,y_vector,XTX)
    x = exp.(x)
    if length(x) == 2
        lml,grad = log_marginal_likelihood_overdetermined(
        Φ,
        y_vector,
        1/x[1],
        1/x[2],
        XTX,
    )
    else
        lml,grad = log_marginal_likelihood_overdetermined(
        Φ,
        y_vector,
        1 ./ x[1:end-1],
        1/x[end],
        XTX,
    )
    end
    return -lml
end
#########################################################################################################################################

end