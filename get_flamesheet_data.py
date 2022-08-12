import cantera as ct
import numpy as np
import sys
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

##### INPUTS #####
pressures = [1.0]
temperatures = [298]
phis = [1.0]
fuel = 'CH4'
oxid = 'O2:1.0, N2:3.76'
mechanism = 'drm19.yaml'
trans = 'Mix'
progvars = ["CO2","H2O","CO","H2"]
ctable_specs = ["CO2","H2O","CO","H2","N2","O2","OH"]
outputdir = "flamesheet"

verbose = False
use_mpi = False

# Flame Numerics
width = 0.1
loglevel = 1
ratio = 2
slope = 0.05
curve = 0.05
prune = 0.02
max_points = 10000

###### SETUP ######
pressures = 101325.0 * np.array(pressures)

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
if use_mpi:
    from  mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
else:
    rank = 0
    nprocs=1
    
cond_iterator_global = list(itertools.product(phis,pressures,temperatures))
cond_iterator = cond_iterator_global[rank::nprocs]

###### RUN #######
for phi,press,temp in cond_iterator:
    label = 'phi{:06.3f}_P{:06.0f}_T{:04.0f}'.format(phi, press, temp)
    print('rank ', rank, ' computing ', label, flush=True)
    
    # Set up the flame
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxid)
    gas.TP = temp, press
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
    flame.set_max_grid_points(1,max_points)
    flame.transport_model = trans

    # Solve Flame
    flame.solve()
    states = flame.to_solution_array()

    # save quantities: T, p, rho, e_int, h_int, rhoD, Y_*, RR_*
    # ALL : convert from MKS (Cantera) to CGS (Pele) where necessary
    savedata = pd.DataFrame()
    savedata['X'] = states.grid
    savedata['T'] = states.T
    savedata['p'] = states.P * 10.0
    savedata['RHO'] = states.density_mass * 1e-3
    savedata['e_int'] = states.int_energy_mass * 1e7 * 1e-3
    savedata['h'] = states.enthalpy_mass * 1e7 * 1e-3
    savedata['DIFF'] = states.thermal_conductivity / states.cp_mass * 1e3 / 100.0 # This is actually rho*D = lambda/cp (assumes Le=1)
    savedata['VISC'] = states.viscosity * 1e3 / 100.0
    savedata = savedata.assign(**dict(zip(['Y-' + spec for spec in states.species_names], states.Y.T)))
    rxn_rates = states.net_production_rates * states.molecular_weights # convert mole basis to mass basis
    rxn_rates[0,:] = rxn_rates[1,:] # overwrite 0th step (otherwise PROG basis will never ignite)
    rxn_rates *= 1e-3 # MKS to CGS conversion
    savedata = savedata.assign(**dict(zip(['SRC_' + spec for spec in states.species_names], rxn_rates.T)))
    print(np.max(np.diff(savedata['Y-CO2']+savedata['Y-CO']+savedata['Y-H2O']+savedata['Y-H2'])))
    print(np.max(np.diff(savedata['T'])))
    savedata.to_csv(os.path.join(outputdir, 'flamesheet_'+label+'.csv'))
