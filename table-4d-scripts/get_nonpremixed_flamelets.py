import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

##### INPUTS #####

# Setup and IO
use_MPI = True
data_directory  = 'new-flames-ig'

# Physical parameters & BCs
press = 300.0 * ct.one_atm
fuel_temp = 343.15
fuel_comp = "CH4:1"
ox_temp = 1005.35
ox_comp = "O2:0.2, CO2:0.8"
dil_temp = 783.15
dil_comp = "CO2:1"
oxidizer_splits = np.linspace(0,1,21)
flame_width = 0.01
mdot_initial = 0.1 # kg/m2/s

# Parameters to compute extinction
delta_temperature_limit_extinction = 50 # k
delta_alpha = 1.0  # initial change in strain rate
delta_alpha_min = 0.0025 # Limit of the refinement: Minimum normalized strain rate increase
delta_alpha_max = 3
delta_T_min = 5  # K # Limit of the Temperature decrease
delta_T_max = 30
delta_alpha_max_change_factor = 3

# Models
mechanism = '/Users/bperry/Software/PelePhysics/Mechanisms/drm19/mechanism.yaml'    # Only this mechanism is compatible with P-R and R-K EOS
transport = 'UnityLewis'    # 'Mix' (mixture-avg diffdiff) or 'UnityLewis'
eos       = 'Peng-Robinson' # 'Peng-Robinson' or 'Redlich-Kwong' or 'ideal-gas'

# Numerical parameters
loglevel = 0
ratio = 2.0
slope = 0.06
curve = 0.06
prune = 0.02
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
    oxstream.mass   = ww
    dilstream.mass  = (1-ww)
    ox_mixture = oxstream + dilstream

    flame = ct.CounterflowDiffusionFlame(gas, width=flame_width)
    flame.P = press
    flame.fuel_inlet.Y = fuelstream.Y
    flame.fuel_inlet.T = fuelstream.T
    flame.fuel_inlet.mdot = mdot_initial
    flame.oxidizer_inlet.Y = ox_mixture.Y
    flame.oxidizer_inlet.T = ox_mixture.T
    flame.oxidizer_inlet.mdot = mdot_initial

    temperature_limit_extinction = max(fuelstream.T, ox_mixture.T) + delta_temperature_limit_extinction

    flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
    flame.flame.set_steady_tolerances(default=tols)
    flame.max_grid_points = maxpoints
    flame.max_time_step_count = maxsteps
    flame.transport_model = transport

    # Initialize and solve
    print('rank {}: Creating the initial solution'.format(rank), flush=True)
    flame.solve(loglevel=loglevel, auto=True)

    file_name = 'initial_solution_'+label+'.yaml'
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
    alpha = [1.]
    # Factor of refinement of the strain rate increase
    delta_alpha_factor = 4.

    # Iteration indicator
    n = 0
    # Indicator of the latest flame still burning
    n_last_burning = 0
    # List of peak temperatures
    T_max = [np.max(flame.T)]
    # List of maximum axial velocity gradients
    a_max = [np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid)))]

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
            flame.write_csv(os.path.join(data_directory, file_name.replace('.yaml','.csv')),
                            quiet=True, species='Y')
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
            flame.write_csv(os.path.join(data_directory, file_name.replace('.yaml','.csv')),
                            quiet=True, species='Y')
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
