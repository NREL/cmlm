import cantera as ct
import numpy as np
import sys
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

##### INPUTS #####
use_mpi = True
verbose = False
T_inc_max = 10.0
t_end = 1e-2
dt_max = 1.0e-4
pressures = [1.0] # atm
temperatures = [1400.0] # K
phis = np.linspace(0.5,1.6,111) # equivalence ratio
mechanism = 'drm19.yaml'
fuel = {'CH4':1.0}
oxidizer = {'o2':1.0, 'n2':3.76}
outputdir = 'autoignition'

###### SETUP ######
pressures = 101325.0 * np.array(pressures)

if not os.path.exists(outputdir): os.makedirs(outputdir)
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

def compute_idt(states, spec):
    idx_ig = np.argmax(states.Y[:,states.species_index(spec)])
    idx_ig = np.argmin(np.abs(states.T - (states.T[0] + states.T[-1])/2))
    return states.t[idx_ig]

###### RUN #######
for phi,press,temp in cond_iterator:
    label = 'phi{:06.3f}_P{:06.0f}_T{:04.0f}'.format(phi, press, temp)
    print('rank ', rank, ' computing ', label, flush=True)
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temp, press
    
    reactor = ct.IdealGasConstPressureReactor(gas)
    states = ct.SolutionArray(gas, extra=['t'])
    states.append(reactor.thermo.state, t=0.0)
    sim = ct.ReactorNet([reactor])
    reactor.set_advance_limit('temperature', T_inc_max)
    reactor.set_advance_limit('CO2', 0.002)
    sim.verbose = verbose
    while sim.time < t_end:
        sim.advance(sim.time + dt_max)
        states.append(reactor.thermo.state, t=sim.time)
    print('rank ', rank, ' finished ', label, ' IDT is ',
          compute_idt(states, 'HO2'), ' Tfinal is ', states.T[-1], ' npoints saved ', len(states.T), flush=True)

    # save quantities: T, p, rho, e_int, h_int, rhoD, Y_*, RR_*
    # ALL : convert from MKS (Cantera) to CGS (Pele) where necessary
    savedata = pd.DataFrame(index=states.t)
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
    savedata.to_csv(os.path.join(outputdir, 'autoig_'+label+'.csv'))
