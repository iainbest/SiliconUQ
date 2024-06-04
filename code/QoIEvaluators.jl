module QoIEvaluators

using ASE, ACE1, JuLIP, IPFitting
using LinearAlgebra, Polynomials, Distributions
using IB_BayesianLinearRegression
using Plots, PyCall

using ..ACE_fs

### initialize python functions and classes
function __init__()
    py"""
    import numpy as np
    from ase.calculators.calculator import Calculator
    from ase.constraints import full_3x3_to_voigt_6_stress
    from ase.constraints import ExpCellFilter

    from ase.build import bulk
    from ase.units import GPa

    from ase.optimize import QuasiNewton
    from ase.mep import NEB              ### changed from neb -> mep
    from ase.mep.neb import NEBOptimizer              ### changed from neb -> mep.neb
    from ase.mep.neb import NEBTools              ### changed from neb -> mep.neb

    from ase.io import read

    from ase.utils.forcecurve import fit_images
    from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

    from julia.api import Julia
    jl = Julia(compiled_modules=False)

    from julia import Main
    Main.eval("using ASE, JuLIP, ACE1")
    Main.eval("using IPFitting")
    Main.eval("function build_ACE_from_coefficients(coeffs,B,species,isolated_atom_energy); Vref = OneBody(Dict(species => isolated_atom_energy)); V = JuLIP.MLIPs.SumIP(Vref,JuLIP.MLIPs.combine(B,coeffs)); return V; end")

    ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
    ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
    convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

    class ACE_UQ_Calculator(Calculator):
    ### ASE-compatible Calculator that calls JuLIP.jl for forces and energy

        implemented_properties = ['forces', 'energy', 'free_energy', 'stress']
        default_parameters = {}
        name = 'JulipCalculator'

        def __init__(self,c,B,species,isolated_atom_energy):
            Calculator.__init__(self)
            self.julip_calculator = Main.build_ACE_from_coefficients(c,B,species,isolated_atom_energy)

        def calculate(self, atoms, properties, system_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            julia_atoms = ASEAtoms(atoms)
            julia_atoms = convert(julia_atoms)
            self.results = {}
            if 'energy' in properties:
                E = Main.energy(self.julip_calculator, julia_atoms)
                self.results['energy'] = E
                self.results['free_energy'] = E
            if 'forces' in properties:
                self.results['forces'] = np.array(Main.forces(self.julip_calculator, julia_atoms))
            if 'stress' in properties:
                voigt_stress = full_3x3_to_voigt_6_stress(np.array(Main.stress(self.julip_calculator, julia_atoms)))
                self.results['stress'] = voigt_stress

    def py_elasticconstants(c,B,species,isolated_atom_energy):

        calc = ACE_UQ_Calculator(c,B,species,isolated_atom_energy)

        at = bulk(species)
        at.calc = calc
        ### attach filter to relax positions and unit cell
        ecf = ExpCellFilter(at)

        ### relax bulk atomic positions and cell - fit_elastic_constants expects this
        qn = QuasiNewton(ecf)
        qn.run(fmax=1e-4,steps=1000)

        C,C_err = fit_elastic_constants(at,'cubic',verbose=False,optimizer=QuasiNewton,fmax=1e-4,steps=1000)

        c11,c12,c44 = Voigt_6x6_to_cubic(C/GPa)

        return c11,c12,c44

    def py_vacancyformationenergy(c,B,species,isolated_atom_energy):

        bm = bulk(species,cubic=True)
        calc = ACE_UQ_Calculator(c,B,species,isolated_atom_energy)
        bm.calc = calc
        ecf = ExpCellFilter(bm)

        f_tol=1e-4 
        ### relax bulk state
        qn = QuasiNewton(ecf, trajectory="_bulk.traj")
        qn.run(fmax=f_tol,steps=1000)

        ### build supercell by repeating bulk in all directions
        bulk_material = bm*(3,3,3)

        ### apply calculator to initial state
        bulk_material.calc = calc

        ### get energy of the bulk state
        e_bulk = bulk_material.get_potential_energy()

        ### Get number of atoms in bulk state
        N = len(bulk_material)

        idx = N//2

        ### get position & index of vacancy
        vacancy_position, vacancy_index = bulk_material[idx].position, bulk_material[idx].index

        ### because of zero indexing, we pop atom labelled vacancy_index
        bulk_material.pop(vacancy_index)

        initial_vacancy = bulk_material.copy()
        initial_vacancy.calc = calc

        f_tol = 1e-4 
        ### relax vacancy
        qn = QuasiNewton(initial_vacancy, trajectory="_initial.traj")
        qn.run(fmax=f_tol,steps=1000)

        if not (qn.converged()):
            print("not converged vacancy")
            return -10000.0

        ### read back in initial and final states, reapply calculator
        initial = read("_initial.traj")
        initial.calc = calc

        ### calculate energy difference (this is a simple vacancy formation energy...)
        norm_energy = initial.get_potential_energy() - (e_bulk * (N-1)/N)

        return norm_energy

    def py_vacancymigration(c,B,species,isolated_atom_energy):

        bm = bulk(species,cubic=True)
        # bm = bulk(species)
        calc = ACE_UQ_Calculator(c,B,species,isolated_atom_energy)
        bm.calc = calc
        ecf = ExpCellFilter(bm)
        f_tol=1e-4

        ### relax bulk state
        qn = QuasiNewton(ecf, trajectory="bulk.traj")
        qn.run(fmax=f_tol,steps=1000)

        ### build supercell by repeating bulk in all directions
        bulk_material = bm*(3,3,3)

        ### apply calculator to initial state
        bulk_material.calc = calc

        ### get energy of the bulk state
        e_bulk = bulk_material.get_potential_energy()

        ### Get number of atoms in bulk state
        N = len(bulk_material)
        idx = N//2

        ### set number of images in NEB (including start/end points)
        n_images = 11

        ### get position & index of vacancy
        vacancy_position, vacancy_index = bulk_material[idx].position, bulk_material[idx].index

        ### because of zero indexing, we pop atom labelled by vacancy_index
        bulk_material.pop(vacancy_index)

        initial_vacancy = bulk_material.copy()
        initial_vacancy.calc = calc

        ### now create final vacancy state
        final_vacancy = initial_vacancy.copy()

        ### we move atom labelled 26 to the vacancy site
        final_vacancy_positions = final_vacancy.get_positions()
        final_vacancy_positions[vacancy_index,:] = vacancy_position

        final_vacancy.set_positions(final_vacancy_positions)

        ### apply calculator to final state
        final_vacancy.calc = calc

        f_tol = 1e-4 
        ### relax initial and final states
        qn = QuasiNewton(initial_vacancy, trajectory="initial.traj")
        qn.run(fmax=f_tol,steps=1000)

        if not (qn.converged()):
            print("not converged initial/final images")
            return (-1.0,np.repeat([-1.0],n_images),np.repeat([-1.0],n_images),
                    np.repeat([-1.0],(n_images-1)*20 +1),np.repeat([-1.0],(n_images-1)*20 +1))

        qn = QuasiNewton(final_vacancy, trajectory="final.traj")
        qn.run(fmax=f_tol,steps=1000)

        ### read back in initial and final states, reapply calculator
        initial = read("initial.traj")
        final = read("final.traj")
        initial.calc = calc
        final.calc = calc

        ### calculate energy for normalisation (this is a simple vacancy formation energy...)
        norm_energy = initial.get_potential_energy() - (e_bulk * (N-1)/N)

        ### build vector of images for NEB
        images = [initial]
        images += [initial.copy() for i in range(n_images-2)]
        images += [final]
        for image in images[1:n_images-1]:
            image.calc = calc

        ### set up and run NEB
        neb = NEB(images,allow_shared_calculator=True)
        neb.set_calculators(calc)
        neb.interpolate()
        qn = NEBOptimizer(neb, trajectory="neb.traj")

        f_tol = 5e-4
        qn.run(fmax=f_tol,steps=1000);

        ### check convergence of neb
        if not (qn.get_residual() < f_tol):
            print("not converged neb")
            return (-1.0,np.repeat([-1.0],n_images),np.repeat([-1.0],n_images),
                    np.repeat([-1.0],(n_images-1)*20 +1),np.repeat([-1.0],(n_images-1)*20 +1))


        ### read in images from trajectory
        images = read("neb.traj@-11:")

        # ### reading in images from trajectory for some reason loses calculator(s) - reapplying them...
        for i in range(0,n_images):
            images[i].calc = calc

        nebtools = NEBTools(images)

        ### Get the calculated barrier and the energy change of the reaction.
        Ef, dE = nebtools.get_barrier()

        ### Create a figure like that coming from ASE-GUI.
        ### includes positions of atoms in images, energies of atoms in images, positions of interpolated fit,... 
        ### ...energies of interpolated fit, and tangent lines
        forcefit = fit_images(images)

        return Ef,forcefit[0],[i+norm_energy for i in forcefit[1]],forcefit[2],[i+norm_energy for i in forcefit[3]]

    """
end

### python imports and definitions
ase = pyimport("ase")
pyimport("ase.optimize")
pyimport("ase.mep")              ### changed from neb -> mep
pyjulip = pyimport("pyjulip")

QuasiNewton = ase.optimize.QuasiNewton
NEB = ase.mep.NEB              ### changed from neb -> mep
NEBOptimizer = ase.mep.neb.NEBOptimizer              ### changed from neb -> neb.mep
NEBTools = ase.mep.neb.NEBTools              ### changed from neb -> mep

export bulk_modulus_from_v_and_e_list, bulkmodulus, elasticconstants, vacancyformationenergy, vacancymigration
export bulkmodulusreduced, elasticconstantsreduced, vacancyformationenergyreduced
export ComplexRootsException
export kJ

const _e = 1.602176634e-19 ### elementary unit charge
const kJ = 1000 / _e;      ### kJ in ASE units -to return bulk mod in sensible units (GPa), multiply by 1e24

struct ComplexRootsException <: Exception
end


"""
    bulk_modulus_from_v_and_e_list(volumes,energies)

Calculates the bulk modulus of a material from its energy-volume curve, given a list of volumes and a list 
of energies. (WIP)

### Arguments

- `volumes::Vector{Float64}`: list of volumes of atomic configurations
- `energies::Vector{Float64}`: list of energies of atomic configurations

Similar to ASE 's EOS function. Performs a cubic polynomial fit, and outputs estimate of bulk modulus from 
curvature information at minimum. Units consistent with ASE, bulk modulus is output in units of GPa.

NOTE: Same SJEOS (10.1103/PhysRevB.63.224115, 10.1103/PhysRevB.67.026103) procedure as ASE 's EOS routine.
"""
function bulk_modulus_from_v_and_e_list(volumes::Vector{Float64},energies::Vector{Float64})::Float64
   
    ### perform polyfit on volumes^(-1/3) and energies
    fit0 = Polynomials.fit(volumes.^(-1/3),energies,3)

    ### calculate derivates
    fit1 = derivative(fit0)
    fit2 = derivative(fit1)

    t_min = nothing
    for t in roots(fit1)
        if (!(typeof(t) <: Real))
            e = ComplexRootsException()
            throw(e)
        elseif (t > 0) & (fit2(t) > 0)
            t_min = t
            break
        end
    end
    
    if t_min === nothing
        return -100/kJ*1.0e24
    end
    
#     v0 = (t_min)^(-3)
    B = (t_min^5)*fit2(t_min)/9
    
    ### return bulk modulus in GPa instead of ASE units ( eV/(Angstrom^3) )
    return B/kJ*1.0e24

end

"""
    bulkmodulus(V,species)

Calculates the bulk modulus of a material from its energy-volume curve

### Arguments

- `V::`: ACE potential
- `species::Symbol`: chemical species of atoms (single species only)
"""
function bulkmodulus(V,species::Symbol)::Float64

    ### generate predictive configs around minimum of E-V curve
    delta = 0.055
    scale_factors_p = (1.0-delta):0.001:(1.0+delta)
    predictive_cfgs = Vector{Atoms{Float64}}(undef,length(scale_factors_p));

    for (count,sf) in enumerate(scale_factors_p)
        ### start with bulk config
        bulk_config = bulk(species,cubic=true)
        
        ### scale unit cell AND atom positions 
        set_cell!(bulk_config, bulk_config.cell * sf)
        set_positions!(bulk_config, bulk_config.X * sf)
        
        ### append to new config list
        predictive_cfgs[count] = bulk_config
    end

    ### calculate volumes of predictive configurations
    predictive_volumes = [volume(predictive_cfgs[i]) for i in 1:length(predictive_cfgs)];

    ### calculate energies of predictive configurations
    predictive_energies = [energy(V,predictive_cfgs[i]) for i in 1:length(predictive_cfgs)];

    Bm = 0.0

    try 
        # evaluate bulk modulus from energy and volume lists
        Bm = bulk_modulus_from_v_and_e_list(predictive_volumes,predictive_energies)

        catch e

        if typeof(e) <: ComplexRootsException
            ### do something
            Bm = -100/kJ*1.0e24
        else
            rethrow()
        end
    end

    return Bm

end

"""
    elasticconstants(coeffs,B,species,isolated_atom_energy)

### Arguments

- `c::`: coefficients of ACE potential
- `B::`: basis of ACE potential
- `species::Symbol`: chemical species of bulk system (single species only)
- `isolated_atom_energy`
"""
function elasticconstants(c,B,species,isolated_atom_energy)

    elastic_constants = py"py_elasticconstants"(c,B,species,isolated_atom_energy)
    return elastic_constants
end

"""
    vacancyformationenergy(c,B,species,isolated_atom_energy)

### Arguments

- `c::`: coefficients of ACE potential
- `B::`: basis of ACE potential
- `species::Symbol`: species of atom
- `isolated_atom_energy`

"""
function vacancyformationenergy(c,B,species,isolated_atom_energy)
    e_vac = py"py_vacancyformationenergy"(c,B,species,isolated_atom_energy)
    return e_vac
end

"""
    vacancymigration(c,B,species,isolated_atom_energy)

Calculates vacancy migration path and associated energy barrier using Nudged Elastic Band (NEB).

### Arguments

- `c::`: coefficients of ACE potential
- `B::`: basis of ACE potential
- `species::Symbol`: species of atom
- `isolated_atom_energy`

"""
function vacancymigration(c,B,species,isolated_atom_energy)
    E_barrier,image_pos,image_E,fit_pos,fit_E = py"py_vacancymigration"(c,B,species,isolated_atom_energy)
    return E_barrier,image_pos,image_E,fit_pos,fit_E
end

function unpackvacancymigrationoutput(vm_output)
    
    obj1 = Vector{Float64}(undef,length(vm_output))
    obj2 = Vector{Vector{Real}}(undef,length(vm_output))
    obj3 = Vector{Vector{Float64}}(undef,length(vm_output))
    obj4 = Vector{Vector{Float64}}(undef,length(vm_output))
    obj5 = Vector{Vector{Float64}}(undef,length(vm_output))
    
    for (count,obj) in enumerate(vm_output)
        obj1[count] = obj[1]
        obj2[count] = obj[2]
        obj3[count] = obj[3]
        obj4[count] = obj[4]
        obj5[count] = obj[5]
    end
    return obj1,obj2,obj3,obj4,obj5
end

"""
    virial_matrix_to_vector(matrix)

Take 3x3 virial stress matrix, return corresponding voigt vector. In matrix notation (row,column), the 
elements 11,22,33,23,13,12 are contained in the vector, in that order.
"""
function virial_matrix_to_vector(matrix::AbstractArray{Float64})
    
    @assert size(matrix) == (3,3)
    
    m_vector = vec(transpose(matrix))
    
    voigt_indices = [1,5,9,6,3,2]
    
    voigt_vector = m_vector[voigt_indices]
    
    return voigt_vector
end

"""
get virials, energies and forces for a vector of configs, output in same format as e.g. design matrix, observation vector
"""
function efv(V,species::Symbol,configs::Vector{Dat})

    efv_vector = Vector{Vector{Float64}}(undef,0)

    for config in configs
        config_obs = Vector{Float64}(undef,0)
        ### names in configs should be ordered in same way - if not we are in trouble
        for name in names(config.D)
            if name == "E"
                push!(config_obs,energy(V,config.at))
            elseif name == "F"
                config_obs = vcat(config_obs,reduce(vcat,forces(V,config.at)))
            elseif name == "V"
                config_obs = vcat(config_obs,vcat(virial_matrix_to_vector(virial(V,config.at))))
            else
                error("unknown observation type?")
            end
        end
        push!(efv_vector,config_obs)
    end

    efv_vector = reduce(vcat,efv_vector)

    return efv_vector

end

########################################################################################################
########################################################################################################
### From JuLIPMaterials.jl https://github.com/cortner/JuLIPMaterials.jl/blob/master/src/CLE.jl . #######

function elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms, h=1e-4)
    X0, cell0 = positions(at), cell(at)
    Ih = Matrix(1.0I(3))
    h === nothing && (h = eps()^(1/3))
    C = zeros(3,3,3,3)
    for i = 1:3, a = 1:3
       set_positions!(at, X0)
       set_cell!(at, cell0)
 
       Ih = Matrix(1.0I(3))
       Ih[i,a] += h
       Ih[a,i] += h
       apply_defm!(at, Ih)
       Sp = stress(calc, at)
 
       Ih[i,a] -= 2h
       Ih[a,i] -= 2h
       apply_defm!(at, Ih)
       Sm = stress(calc, at)
 
       C[i, a, :, :] = (Sp - Sm) / (2*h)
    end
    set_positions!(at, X0)
    set_cell!(at, cell0)
    # symmetrise it - major symmetries C_{iajb} = C_{jbia}
    for i = 1:3, a = 1:3, j=1:3, b=1:3
       C[i,a,j,b] = C[j,b,i,a] = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
    end
    # minor symmetries - C_{iajb} = C_{iabj}
    for i = 1:3, a = 1:3, j=1:3, b=1:3
       C[i,a,j,b] = C[i,a,b,j] = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
    end
    return C
 end

"""
Adding relaxation step after deformations since Si diamond unit cell has two atoms instead of one 

"""
function elastic_moduli_(calc::AbstractCalculator, at::AbstractAtoms, h=1e-4)
    
   ### ADDED
   set_calculator!(at,calc)
    
   X0, cell0 = positions(at), cell(at)
   Ih = Matrix(1.0I(3))
   h === nothing && (h = eps()^(1/3))
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      set_positions!(at, X0)
      set_cell!(at, cell0) 

      Ih = Matrix(1.0I(3))
      Ih[i,a] += h
      Ih[a,i] += h
      apply_defm!(at, Ih)
        
      ### ADDED  
      fixedcell!(at)
      minimise!(at,verbose=0)
      variablecell!(at)  
        
      Sp = stress(calc, at)
        
      ### ADDED          
      set_positions!(at,X0)
      set_cell!(at,cell0)
        
      Ih[i,a] -= h
      Ih[a,i] -= h
        
      ### COMMENTED OUT
#       Ih[i,a] -= 2h
#       Ih[a,i] -= 2h

      apply_defm!(at, Ih)
        
      ### ADDED  
      fixedcell!(at)
      minimise!(at,verbose=0)
      variablecell!(at)   
        
      Sm = stress(calc, at)

      C[i, a, :, :] = (Sp - Sm) / (2*h)
   end
   set_positions!(at, X0)
   set_cell!(at, cell0)
   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[j,b,i,a] = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[i,a,b,j] = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
   end
   return C
end

const voigtinds = [1, 5, 9, 8, 7, 4] # xx yy zz yz xz xy

voigt_moduli(C::Array{T,4}) where {T} = reshape(C, 9, 9)[voigtinds, voigtinds]

function cubic_moduli(C::AbstractMatrix; tol=1e-4)
   C = copy(C)
   idxss = [[(1,1), (2,2), (3,3)],                       # C11
            [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)],  # C12 
            [(4,4), (5,5), (6,6)]]                       # C44

   cubic_C = []
   for idxs in idxss
      Cs = view(C, CartesianIndex.(idxs))
      @assert maximum(abs.(Cs .- mean(Cs))) < tol
      push!(cubic_C, mean(Cs))
      fill!(Cs, 0.0)
   end
   # check remaining elements are near to zero
   # @assert maximum(abs.(C)) < tol 
   return Tuple(cubic_C)
end

########################################################################################################
########################################################################################################
########################################################################################################

### Reduced functions used to aim to take in configs and output posterior predictive QoIs (hence some hardcoding or nonsense defs)
### Only use now is to get output type of QoI, can otherwise ignore - these are almost entirely unused.

bulkmodulusreduced(V,species) = bulkmodulus(v,species)

### Function which takes sample from  predictive features and predictive configurations (Y_p_s and predictive_at) and returns (relaxed) VFE.

function vacancyformationenergyreduced(Y_p_s,predictive_at,isolated_atom_energy)::Float64
    
    ### account for isolated_atom_energy in energies
    ### remove hardcoding of relevant values at some point
    Y_p_s[662] += length(predictive_at[2])*isolated_atom_energy

    e_bulk_per_atom = Y_p_s[7] / length(predictive_at[1]) + isolated_atom_energy

    VFE = Y_p_s[662] - (e_bulk_per_atom*length(predictive_at[2]))

    return VFE
end

function elasticconstantsreduced(c,B,species,isolated_atom_energy)::Tuple{Float64,Float64,Float64}
    return (0.0,0.0,0.0)
end

end