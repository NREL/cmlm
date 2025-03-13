import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

##### INPUTS #####

# Setup and IO
use_MPI = False
data_directory  = 'nonpremixed_flames'

# Physical parameters & BCs
press = 1.0 * ct.one_atm
fuel_temp = 300.0
fuel_comp = "CH4:1"
ox_temp = 300.0
ox_comp = "O2:1.0, N2:3.76"
dil_temp = 300.0
dil_comp = "O2:1.0, N2:3.76"

# Settings for the chemtable
oxidizer_splits = [1.0]

# Parameters to compute extinction
mdot_initial = 0.1 # kg/m2/s
delta_temperature_limit_extinction = 50 # k
delta_alpha = 1.0  # initial change in strain rate
delta_alpha_min = 0.0025 # Limit of the refinement: Minimum normalized strain rate increase
delta_alpha_max = 3
delta_T_min = 5  # K # Limit of the Temperature decrease
delta_T_max = 20
delta_alpha_max_change_factor = 3

# Models
#mechanism = 'grimech30-noArN.yaml'
mechanism = 'drm19.yaml'
transport = 'UnityLewis'    # 'Mix' (mixture-avg diffdiff) or 'UnityLewis'
eos       = 'gas'

# Numerical parameters
flame_width = 1.0
loglevel = 1
ratio = 3.0
slope = 0.15
curve = 0.15
prune = 0.05
tols = [1e-6, 1e-12]
initpoints = 50
maxpoints = 8192
maxsteps = 4096
printint = -1
saveint = 10

#### RUN ####

# set up the streams
ox = ct.Solution(mechanism, name=eos)
ox.TPY = ox_temp, press, ox_comp
oxstream = ct.Quantity(ox, constant='HP')
fuel = ct.Solution(mechanism, name=eos)
fuel.TPY = fuel_temp, press, fuel_comp
fuelstream = ct.Quantity(fuel, constant='HP')
dil = ct.Solution(mechanism, name=eos)
dil.TPY = dil_temp, press, dil_comp
dilstream = ct.Quantity(dil, constant='HP')
gas = ct.Solution(mechanism, name=eos)

# Set up the cases to be run
cond_iterator_global = list(itertools.product(oxidizer_splits))
data = {}
labels={}
if use_MPI:
    from  mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
else:
    rank = 0
    nprocs = 1
cond_iterator = cond_iterator_global[rank::nprocs]
for ww, in cond_iterator_global:
    label = 'W{:06.4f}'.format(ww)
    labels[(ww,)] = label
    if (ww,) in cond_iterator:
        data[label] = {'P':press, 'W':ww}

if rank ==0:
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

# save full csv data
def save_csv(flame, filename):
    savedata = pd.DataFrame(index=flame.grid)
    savedata['Zmix'] = flame.mixture_fraction(m='N')
    savedata['T'] = flame.T
    savedata['p'] = flame.P * 10.0
    savedata['RHO'] = flame.density_mass * 1e-3
    savedata['e_int'] = flame.int_energy_mass * 1e7 * 1e-3
    savedata['h'] = flame.enthalpy_mass * 1e7 * 1e-3
    savedata['DIFF'] = flame.thermal_conductivity / flame.cp_mass * 1e3 / 100.0 # This is actually rho*D = lambda/cp (assumes Le=1)
    savedata['VISC'] = flame.viscosity * 1e3 / 100.0
    savedata = savedata.assign(**dict(zip(['Y-' + spec for spec in flame.gas.species_names], flame.Y)))
    rxn_rates = flame.net_production_rates.T * flame.gas.molecular_weights # convert mole basis to mass basis
    #rxn_rates[0,:] = 0.0; rxn_rates[-1,:] = 0.0 # No rection at Z=0 or Z=1
    rxn_rates *= 1e-3 # MKS to CGS conversion
    savedata = savedata.assign(**dict(zip(['SRC_' + spec for spec in flame.gas.species_names], rxn_rates.T)))
    savedata.to_csv(filename)

# Convergence criteria for extinction
def time_to_stop(Tmax, dTmin, dalpha, dalphamin):
    if len(Tmax) < 2: # stop if first flame is extinguished
        return True
    #elif ( (Tmax[-2] - Tmax[-1]) > dTmin):  # Temp not converged
    #    return False
    elif (dalpha > dalphamin): # strain rate not converged
        return False
    return True

# Laminar Flame Calculations
for ww, in cond_iterator:
    label = labels[(ww,)]

    # Set up flame
    fuelstream.moles   = ww
    dilstream.moles  = (1-ww)
    fuel_mixture = fuelstream + dilstream

    flame = ct.CounterflowDiffusionFlame(gas, width=flame_width)
    flame.P = press
    flame.fuel_inlet.Y = fuel_mixture.Y
    flame.fuel_inlet.T = fuel_mixture.T
    flame.fuel_inlet.mdot = mdot_initial
    flame.oxidizer_inlet.Y = oxstream.Y
    flame.oxidizer_inlet.T = oxstream.T
    flame.oxidizer_inlet.mdot = mdot_initial

    temperature_limit_extinction = max(fuel_mixture.T, oxstream.T) + delta_temperature_limit_extinction

    flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
    flame.flame.set_steady_tolerances(default=tols)
    flame.max_grid_points = maxpoints
    flame.max_time_step_count = maxsteps
    flame.transport_model = transport

    # Initialize and solve
    print('rank {}: Creating the initial solution'.format(rank), flush=True)
    flame.solve(loglevel=loglevel, auto=True)

    n_init = 100
    n = n_init
    n_last_burning = n
    file_name = 'extinction_'+label+'_{0:04d}.yaml'.format(n)
    flame.save(os.path.join(data_directory, file_name), name='solution_'+label,
           description='Cantera version ' + ct.__version__ +
               ', reaction mechanism ' + mechanism + ', equation of state ' + eos)

    # PART 2: COMPUTE EXTINCTION STRAIN
    # from: https://cantera.org/examples/python/onedim/diffusion_flame_extinction.py.html

    # Exponents for the initial solution variation with changes in strain rate
    # Taken from Fiala and Sattelmayer (2014)
    exp_d_a = - 1. / 2.
    exp_u_a = 1. / 2.
    exp_V_a = 1.
    exp_lam_a = 2.
    exp_mdot_a = 1. / 2.

    # Set normalized initial strain rate
    alpha = [np.nan] * n_init + [1.]
    # Factor of refinement of the strain rate increase
    delta_alpha_factor = 4.

    # List of peak temperatures
    T_max = [np.nan] * n_init + [np.max(flame.T)]
    # List of maximum axial velocity gradients
    a_max = [np.nan] * n_init + [np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid)))]

    # Simulate counterflow flames at increasing strain rates until the flame is
    # extinguished. To achieve a fast simulation, an initial coarse strain rate
    # increase is set. This increase is reduced after an extinction event and
    # the simulation is again started based on the last burning solution.
    # The extinction point is considered to be reached if the abortion criteria
    # on strain rate increase and peak temperature decrease are fulfilled.
    while True:
        n += 1
        # Update relative strain rates
        alpha.append(alpha[n_last_burning] * (1+delta_alpha))
        strain_factor = alpha[-1] / alpha[n_last_burning]
        # Create an initial guess based on the previous solution
        # Update grid
        flame.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = flame.grid / (flame.grid[-1] - flame.grid[0])
        # Update mass fluxes
        flame.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
        flame.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
        # Update velocities
        flame.set_profile('velocity', normalized_grid,
                      flame.velocity * strain_factor ** exp_u_a)
        flame.set_profile('spread_rate', normalized_grid,
                      flame.spread_rate * strain_factor ** exp_V_a)
        # Update pressure curvature
        flame.set_profile('lambda', normalized_grid, flame.L * strain_factor ** exp_lam_a)
        try:
            flame.solve(loglevel=loglevel)
        except ct.CanteraError as e:
            print('rank {}: Error: Did not converge at n ='.format(rank), n, e, flush=True)
        if np.max(flame.T) > temperature_limit_extinction:
            # Flame is still burning, so proceed to next strain rate
            n_last_burning = n
            file_name = 'extinction_'+label+'_{0:04d}.yaml'.format(n)
            flame.save(os.path.join(data_directory, file_name),
                       name='solution_'+label, loglevel=0,
                       description='Cantera version ' + ct.__version__ +
                       ', reaction mechanism ' + mechanism + ', equation of state ' + eos)
            save_csv(flame, os.path.join(data_directory, file_name.replace('.yaml','.csv')))
            T_max.append(np.max(flame.T))
            a_max.append(np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid))))
            delta_alpha = max(min(delta_alpha*max(min(delta_T_max/(T_max[-2] - T_max[-1]),
                                                      delta_alpha_max_change_factor),
                                                  1/delta_alpha_max_change_factor),
                                  delta_alpha_max),
                              delta_alpha_min)
            print('rank {}: Flame burning at alpha = {:8.4F} with Tmax = {:06.1F}. Ngrid = {:5d} '
                  'Proceeding to the next iteration, '
                  'with delta_alpha = {}'.format(rank,alpha[-1], T_max[-1],len(flame.grid), delta_alpha), flush=True)
        elif (time_to_stop(T_max, delta_T_min, delta_alpha, delta_alpha_min)):
            # If the temperature difference is too small and the minimum relative
            # strain rate increase is reached, save the last, non-burning, solution
            # to the output file and break the loop
            T_max.append(np.max(flame.T))
            a_max.append(np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid))))
            file_name = 'extinction_'+label+'_final_{0:04d}.yaml'.format(n)
            flame.save(os.path.join(data_directory, file_name), name='solution_'+label, loglevel=0)
            save_csv(flame, os.path.join(data_directory, file_name.replace('.yaml','.csv')))
            print('rank {0:}: Flame extinguished at alpha = {1:8.4F}.'.format(rank,alpha[-1]),
                  'Abortion criterion satisfied.')
            break
        else:
            # Procedure if flame extinguished but abortion criterion is not satisfied
            # Reduce relative strain rate increase
            delta_alpha = max(delta_alpha / delta_alpha_factor, delta_alpha_min)
            delta_T_max = max(delta_T_max / delta_alpha_factor, delta_T_min)

            print('rank {0:}: Flame extinguished at alpha = {1:8.4F}. Restoring alpha = {2:8.4F} and '
                  'trying delta_alpha = {3}'.format(
                      rank, alpha[-1], alpha[n_last_burning], delta_alpha))

            # Restore last burning solution
            file_name = 'extinction_'+label+'_{0:04d}.yaml'.format(n_last_burning)
            flame.restore(os.path.join(data_directory, file_name),
                          name='solution_'+label, loglevel=0)

    pd.DataFrame({'a_max':a_max, 'T_max':T_max}).to_csv(data_directory + '/scurve_info_'+label)

    # Traverse the S-curve in the opposite direction
    # Reload initial flame solution
    file_name = 'extinction_'+label+'_{0:04d}.yaml'.format(n_init)
    flame.restore(os.path.join(data_directory, file_name),
                  name='solution_'+label, loglevel=0)
    n = n_init
    while True:
        n -= 1
        print()
        print('HERE                                 WOOHOO ', n, alpha, delta_alpha)
        print()
        # Update relative strain rates
        delta_alpha = 0.4
        strain_factor = 1 / (1+delta_alpha)
        alpha[n] = alpha[n+1] * strain_factor
        # Create an initial guess based on the previous solution
        # Update grid
        flame.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = flame.grid / (flame.grid[-1] - flame.grid[0])
        # Update mass fluxes
        flame.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
        flame.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
        # Update velocities
        flame.set_profile('velocity', normalized_grid,
                      flame.velocity * strain_factor ** exp_u_a)
        flame.set_profile('spread_rate', normalized_grid,
                      flame.spread_rate * strain_factor ** exp_V_a)
        # Update pressure curvature
        flame.set_profile('lambda', normalized_grid, flame.L * strain_factor ** exp_lam_a)
        try:
            flame.solve(loglevel=loglevel)
        except ct.CanteraError as e:
            print('rank {}: Error: Did not converge at n ='.format(rank), n, e, flush=True)

        file_name = 'extinction_'+label+'_{0:04d}.yaml'.format(n)
        flame.save(os.path.join(data_directory, file_name),
                   name='solution_'+label, loglevel=0,
                   description='Cantera version ' + ct.__version__ +
                   ', reaction mechanism ' + mechanism + ', equation of state ' + eos)
        save_csv(flame, os.path.join(data_directory, file_name.replace('.yaml','.csv')))
        T_max[n]= np.max(flame.T)
        a_max[n]= np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid)))
        #if T_max[n] - T_max[n+1] < delta_T_min:
        #    print('Reached Equilibrium at T: ', T_max[n],  T_max[n+1] )
        #    break
        #else:
        #    print('Keeping going at T: ', T_max[n],  T_max[n+1] )
        #    delta_alpha = np.sqrt(2) #max(min(delta_alpha*max(min(delta_T_max/(T_max[n] - T_max[n+1]),
        #                  #                            delta_alpha_max_change_factor),
        #                  #                        1/delta_alpha_max_change_factor),
        #                  #        delta_alpha_max),
        #                  #    delta_alpha_min)


        pd.DataFrame({'a_max':a_max, 'T_max':T_max}).to_csv(data_directory + '/scurve_info_'+label)
