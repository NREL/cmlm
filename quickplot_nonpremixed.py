import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

#plt.rcParams['backend'] = 'qt4agg'

files = sorted(glob.glob('nonpremixed_flames/ext*csv'))
varlist = ['T']
for var in varlist:
    plt.figure(var)
    plt.clf()
    for fi in files:
        data = pd.read_csv(fi)
        plt.plot(data['Zmix'],data[var])

plt.figure('prog')
for fi in files:
    data = pd.read_csv(fi)
    plt.plot(data['Zmix'],data['Y-CO2']+data['Y-CO']+data['Y-H2O']+data['Y-H2'])

plt.show()
