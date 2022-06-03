
# coding: utf-8

"""
Simulate two counter-flow jets of reactants shooting into each other. This
simulation differs from the similar premixed_counterflow_flame.py example as the
latter simulates a jet of reactants shooting into products.
"""

import cantera as ct
import numpy as np
import sys
import os
import itertools
import pandas as pd

##### INPUTS #####
use_mpi = True
pressures = [1.0] #[0.9,1.0,1.1] # atm
temperatures = [300.0] #np.linspace(270,360,4) # K
phis = [1.0] # equivalence ratio
velocities = np.logspace(3,-2,101) # m/s
#velocities = [0]
#velocities = [300]
#velocities = [0.94854948, 9.48549482, 94.85494823]
mechanism = 'jp8-lu-skel-38spec.cti'
fuel = 'POSF10325'
oxidizer = {'o2':1.0, 'n2':3.76}
transport = 'Mix'
outputdir = 'strained_flames'
if not os.path.exists(outputdir): os.makedirs(outputdir)
dataoutdir = 'data'

# Cantera
premwidth = 0.1
loglevel = 0
ratio = 4
slope = 0.1
curve = 0.1
prune = 0.02
initpoints = 50
maxpoints = 8192
maxsteps = 4096
tols = [1e-6, 1e-12]

#### FUNCTIONS ######

# Differentiation function for data that has variable grid spacing Used here to
# compute normal strain-rate
def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)
    dx = np.diff(x)
    dy = np.diff(y)
    dydx[0:-1] = dy/dx
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dydx

def calc_flame_scales(flame,datadict):
    flame_x = flame.flame.grid
    flame_T = flame.T
    dTdx = np.diff(flame_T) / np.diff(flame_x)
    datadict['Tmax'] = max(flame_T)
    datadict['l_f'] = ((max(flame_T)-min(flame_T))/max(dTdx))
    datadict['s_L'] = flame.velocity[0]
    datadict['tau_f'] = datadict['l_f']/datadict['s_L']
    return 0

def computeStrainRates(oppFlame):
    # Compute the derivative of axial velocity to obtain normal strain rate
    strainRates = derivative(oppFlame.grid, oppFlame.velocity)

    # Obtain the location of the max. strain rate upstream of the pre-heat zone.
    # This is the characteristic strain rate
    maxStrLocation = abs(strainRates).argmax()
    minVelocityPoint = oppFlame.velocity[:maxStrLocation].argmin()

    # Characteristic Strain Rate = K
    strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
    K = abs(strainRates[strainRatePoint])

    return strainRates, strainRatePoint, K

def computeConsumptionSpeed(oppFlame):

    Tb = max(oppFlame.T)
    Tu = min(oppFlame.T)
    rho_u = max(oppFlame.density)
    integrand = oppFlame.heat_release_rate/oppFlame.cp
    I = np.trapz(integrand, oppFlame.grid)
    Sc = I/(Tb - Tu)/rho_u
    return Sc

# This function is called to run the solver
def solveOpposedFlame(oppFlame, massFlux=0.12, loglevel=1,
                      ratio=2, slope=0.3, curve=0.3, prune=0.05):
    """
    Execute this function to run the Oppposed Flow Simulation This function
   takes a CounterFlowTwinPremixedFlame object as the first argument
    """

    oppFlame.reactants.mdot = massFlux
    oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)

    if loglevel >= 1: oppFlame.show_solution()
    oppFlame.solve(loglevel, auto=True)

    # Compute the strain rate, just before the flame. This is not necessarily
    # the maximum We use the max. strain rate just upstream of the pre-heat zone
    # as this is the strain rate that computations comprare against, like when
    # plotting Su vs. K
    strainRates, strainRatePoint, K = computeStrainRates(oppFlame)

    return np.max(oppFlame.T), K, strainRatePoint

# Setup MPI
gas = ct.Solution(mechanism)
pressures = 101325.0 * np.array(pressures)
cond_iterator_global = list(itertools.product(velocities,phis,pressures,temperatures))
data = {}
labels={}
if use_mpi:
    from  mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
else:
    rank = 0
    nprocs=1
cond_iterator = cond_iterator_global[rank::nprocs]

for vel,phi,press,temp in cond_iterator_global:
    label = 'phi{:06.3f}_P{:06.0f}_T{:04.0f}_s{:07.2f}'.format(phi, press, temp, vel/premwidth)
    labels[(vel,phi,press,temp)] = label
    #if (vel,phi,press,temp) in cond_iterator:
    #    data[label] = {'phi':phi,'P':press,'T':temp,'vel':vel,'width':premwidth}

# Run the flame calculations
for vel,phi,press,temp in cond_iterator:
    label = labels[(vel,phi,press,temp)]
    print(label, flush=True)
    
    # Create flame and gas objects with appropriate parameters
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temp, press # Convert Pressure atm to Pa

    if vel < 1e-14:
        oppFlame = ct.FreeFlame(gas, grid = np.linspace(0,premwidth,initpoints))
        oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
        oppFlame.flame.set_steady_tolerances(default=tols)
        oppFlame.max_grid_points = maxpoints
        oppFlame.max_time_step_count = maxsteps
        oppFlame.transport_model = transport
        oppFlame.solve(loglevel=loglevel, auto=True)
        data = {'phi':phi,'P':press,'T':temp,'vel':vel,'width':premwidth}
        calc_flame_scales(oppFlame, data)
        T = data['Tmax']
        print(label,"Peak temperature: {0:.1f} K".format(T), flush=True)
        print(label,"Flame Speed: {0:.2f} cm/s".format(data['s_L']*100), flush=True)
        print(label,"Flame Thickness: {0:.2f} mm".format(data['l_f']*1000), flush=True)
    else:
        massFlux = gas.density * vel # units kg/m2/s
        #oppFlame = ct.CounterflowTwinPremixedFlame(gas, grid = np.linspace(0,premwidth,initpoints))
        oppFlame = ct.CounterflowTwinPremixedFlame(gas, width = premwidth)
        oppFlame.transport_model = transport

        # Now run the solver. The solver returns the peak temperature, strain rate and
        # the point which we ascribe to the characteristic strain rate.
        (T, K, strainRatePoint) = solveOpposedFlame(oppFlame, massFlux, loglevel=loglevel,
                                                    ratio=ratio, slope=slope, curve=curve, prune=prune)
        Sc = computeConsumptionSpeed(oppFlame)
        data = {'phi':phi,'P':press,'T':temp,'vel':vel,'width':premwidth}
        data['Sc'] = Sc
        data['K'] = K
        data['xK'] = strainRatePoint
        data['Tmax'] =T

        print(label,"Peak temperature: {0:.1f} K".format(T), flush=True)
        print(label,"Strain Rate: {0:.1f} 1/s".format(K), flush=True)
        print(label,"Consumption Speed: {0:.2f} cm/s".format(Sc*100), flush=True)
        
    csvname = outputdir + '/strained_' + fuel + '_' + label +'.csv'
    if T > 600:
        oppFlame.write_csv(csvname, quiet=True, species='Y')
        # Add reaction rates and internal energy to csv
        csvdata = pd.read_csv(csvname)
        #colnames = ['prod_' + colname + ' (kg/m3.s)' for colname in oppFlame.gas.species_names]
        #rates = np.multiply(oppFlame.gas.molecular_weights.reshape(-1,1) , oppFlame.net_production_rates)
        colnames = ['RR_' + colname for colname in oppFlame.gas.species_names]
        rates = oppFlame.net_production_rates
        csvdata = csvdata.assign(**dict(zip(colnames,rates)))
        eint = oppFlame.int_energy_mass
        csvdata.insert(loc=5, column='eint (J/kg)', value=eint)
        csvdata.to_csv(csvname, index=False)

    pd.DataFrame(data,index=[label]).to_csv(csvname.replace(outputdir,dataoutdir))

