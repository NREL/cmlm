import cantera as ct
import ctable_tools as ctt
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta



##### INPUTS #####

data_directory  = 'flames-rk'
file_pattern = 'extinction_W*yaml'
output_dir = 'ctable-ver2-' + data_directory
output_filename = 'table.ctb'
# Note: all files with the above name pattern will be read
# files will be split by the W term in the filename
# files are expected to have Z be monotonically **decreasing**

mechanism = 'ucf16.yaml'    # Only this mechanism is compatible with P-R and R-K EOS
eos       = 'Redlich-Kwong' # 'Peng-Robinson' or 'Redlich-Kwong' or 'ideal-gas'

nzgrid = 25
Zgrid = list(np.linspace(0,0.08,nzgrid)) + list(np.linspace(0.08,0.24,nzgrid)[1:]) + list(np.linspace(0.24,1,nzgrid)[1:])
#Zgrid = [0, 0.0551538, 1]
Zvargrid = [0.0, 0.25] #
#Zvargrid = np.linspace(0,0.25,26)
Cgrid = np.linspace(0,0.17,40)
prog_definition = {"H2O":1, "CO":-0.3, "CO2":0, "OH":0.0}
#prog_definition = {"H2O":1, "CO":0, "CO2":0, "OH":0.0}
table_species   = ["H2O", "CO", "CO2", "OH", "O2", "CH4"] # species to include in table
table_vars = ["RHO", "T", "DIFF", "VISC", "PROG", "SRC_PROG"] + table_species
pspecs = ["H2O", "CO", "OH"]

##### Get some stuff ready #####
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir)

##### Handy Functions ######

def compute_prog(flame):
    prog = np.zeros(flame.T.shape)
    for spec in prog_definition.keys():
        prog += prog_definition[spec] * flame.Y[gas.species_index(spec)]
    return prog

def compute_progsrc(flame):
    # SRC term (units of 1/s): omega_k = mdot_k / density
    progsrc = np.zeros(flame.T.shape)
    for spec in prog_definition.keys():
        ispec = gas.species_index(spec)
        srcspec = flame.net_production_rates[ispec] * gas.molecular_weights[ispec]
        progsrc += prog_definition[spec] * srcspec
    return progsrc / flame.density

def get_idx_from_fname(fname):
    return int(fname.split('_')[-1].split('.')[0])

def colormap(data):
    return plt.cm.jet(float(data/50))

##### Run ######
# Sort files by their value of W
filenames = sorted(glob.glob(data_directory + '/' + file_pattern))
file_wvals = [(fi[fi.index('W')+1:fi.index('W')+7]) for fi in filenames]
WvalsLabels = sorted(list(set(file_wvals)))
Wvals = [float(Wval) for Wval in WvalsLabels]
WvalsFiles = {Wval:[fi for fi,wfi in zip(filenames,file_wvals) if wfi==Wval]
              for Wval in WvalsLabels}

# Container for final data
finalData = pd.DataFrame(index=pd.MultiIndex.from_product([Wvals,Zgrid,Zvargrid,Cgrid],
                                                         names=['WMIX','ZMIX','ZVAR','PROG']),
                         columns=table_vars, dtype=np.float64)

for wval, wlabel in zip(Wvals, WvalsLabels):

    finames = WvalsFiles[wlabel]
    filtdata = []
    filtfiles = []
    exclusioncount = 0

    print('\n\nFilling table for W = ' + wlabel)

    for flamefile in finames:
        print('Loading and processing file: {}'.format(flamefile))
        gas = ct.Solution(mechanism, eos)
        solname = 'solution_' + flamefile[flamefile.index('W'):flamefile.index('W')+7]
        flame = ct.CounterflowDiffusionFlame(gas)
        flame.restore(flamefile, name=solname, loglevel=10)
        rawdata = pd.DataFrame({'ZMIX':flame.mixture_fraction(m='H'),
                                'PROG':compute_prog(flame),
                                'T':flame.T,
                                'SRC_PROG':compute_progsrc(flame),
                                'DIFF':flame.thermal_conductivity/flame.cp,
                                'VISC':flame.viscosity,
                                'RHO':flame.density})
        for spec in table_species:
            rawdata[spec] = flame.Y[gas.species_index(spec)]

        # Zero source term for first and last flamelet to prevent bad behavior if out of table bounds
        if ('final' in flamefile.split('/')[-1]) or (get_idx_from_fname(flamefile) ==0):
            rawdata['SRC_PROG'] = 0.0

        convdata =  pd.DataFrame(index=pd.MultiIndex.from_product([Zvargrid,Zgrid],
                                                             names=['ZVAR','ZMIX']),
                                 columns=table_vars)
        plt.figure(wlabel + '-all')
        plt.plot(rawdata['ZMIX'].to_numpy(),rawdata['PROG'].to_numpy(),'-',color=colormap(get_idx_from_fname(flamefile)))
        plt.figure(wlabel + '-discard')
        plt.plot(rawdata['ZMIX'].to_numpy(),rawdata['PROG'].to_numpy(),'-',color=colormap(get_idx_from_fname(flamefile)))
        for spec in pspecs:
            plt.figure(wlabel + '-' + spec)
            plt.plot(rawdata['ZMIX'].to_numpy(),rawdata[spec].to_numpy(),'-',color=colormap(get_idx_from_fname(flamefile)))
        plt.figure(wlabel + '-COcolor')
        plt.scatter(rawdata['ZMIX'].to_numpy(),rawdata['PROG'].to_numpy(),s=0.6,c=rawdata['CO'].to_numpy(), vmin=0.0, vmax=0.18)

        # convolute
        # it is assumed that Zmix is decreasing in file
        Zmix = np.array(rawdata['ZMIX'])
        Zmixmid = np.insert([1.0,0.0], 1, 0.5*(Zmix[:-1] + Zmix[1:]))

        # TODO: ordering of these loops for performance?
        for Zvar in Zvargrid:
            for Zval in Zgrid:
                Pdz = np.zeros(Zmix.shape)
                if Zval == 0.0:
                    # Dont set up beta, just take boundary point from flame
                    Pdz[-1] = 1.0
                elif Zval == 1.0:
                    #Just take take boundary point from flame
                    Pdz[0] = 1.0
                elif Zvar == 0.0:
                    # Delta limit of Beta: just do interpolation to Zval
                    idx = np.searchsorted(1.0 - Zmix,1.0-Zval)
                    alpha = (Zval - Zmix[idx-1]) / (Zmix[idx] - Zmix[idx-1])
                    Pdz[idx] = alpha
                    Pdz[idx-1] = 1.0 - alpha
                elif Zvar > Zval*(1-Zval):
                    # Double delta limit: take both boundary points
                    Pdz[0] = Zval
                    Pdz[-1] = 1.0 - Zval
                else:
                    # Just a normal beta distribution
                    a = -Zval * (Zvar + Zval**2 - Zval) / Zvar
                    b = -(1-Zval) * (Zvar + Zval**2 - Zval) / Zvar
                    betadist = beta(a,b)
                    # Get weights for Zvalues in the flamelet
                    cdf = betadist.cdf(Zmixmid)
                    Pdz = cdf[:-1] - cdf[1:]

                # Need to fix convolution for non-density-weighted variables
                convdata.loc[Zvar,Zval] = rawdata[table_vars].mul(Pdz,axis='index').sum()

        # if not the first file being read, ensure monotonicity in progress variable
        save_file = True
        if len(filtdata) > 0:
            nonmonopoints = convdata.loc[0.0,:]['PROG'] > filtdata[-1].loc[0.0,:]['PROG']
            nnonmono = sum(nonmonopoints)
            if nnonmono > 0:
                print('    -> excluding due to non montonicity at {} points'.format(nnonmono))
                save_file = False
                exclusioncount +=1
                plt.figure(wlabel + '-discard')
                tmpdata = convdata.loc[0.0,:]['PROG']
                plt.plot(tmpdata.index, tmpdata.to_numpy(),'r:')
                plt.plot(tmpdata.index[nonmonopoints], tmpdata.to_numpy()[nonmonopoints],'rx')
                print(tmpdata.index[nonmonopoints])


        if save_file:
            plt.figure(wlabel + '-kept')
            tmpdata = convdata.loc[0.0,:]['PROG']
            plt.plot(tmpdata.index, tmpdata.to_numpy(),'-o',color=colormap(get_idx_from_fname(flamefile)))
            filtdata.append(convdata)
            filtfiles.append(flamefile)

    #print(filtdata)
    print('Excluded {} files for nonmontonicity'.format(exclusioncount))
    for suffix in ['all','discard','kept','COcolor'] + pspecs:
        plt.figure(wlabel + '-' + suffix)
        plt.xlabel('Z')
        plt.ylabel('C')
        #plt.xlim([-0.01,0.08])
        #plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'plot_ZC_'+wlabel+'-'+suffix+'.png'))
        plt.clf()
        plt.close()

    #interpolate
    n_cvals = len(filtfiles)
    filtdata = list(reversed(filtdata))
    filtfiles = list(reversed(filtfiles))
    print('Interpolating onto C grid... ')
    for Zval in Zgrid:
        for Zvar in Zvargrid:

            #interpolate between files that were read in based on progress variable

            if n_cvals < 2:
                # if only one progress variable value is available just use that data
                for Cval in Cgrid:
                    finalData.loc[wval, Zval, Zvar, Cval] = filtdata[0].loc[Zvar, Zval]
            else:
                # interpolate
                cvals_loc = [float(fdata.loc[Zvar, Zval]['PROG']) for fdata in filtdata]
                for Cval in Cgrid:
                    right_idx = np.searchsorted(cvals_loc, Cval)
                    if right_idx == 0:
                        # below minimum available C, take minimum
                        left_idx = 0
                        right_idx = 1
                        alpha = 0.0
                    elif right_idx == n_cvals:
                        # above maximum available C, take maximum
                        left_idx = n_cvals -2
                        right_idx = n_cvals -1
                        alpha = 1.0
                    else :
                        left_idx = right_idx -1
                        alpha = (Cval - cvals_loc[left_idx]) / (cvals_loc[right_idx] - cvals_loc[left_idx])

                    finalData.loc[wval, Zval, Zvar, Cval] = ((1.0 - alpha)*filtdata[left_idx].loc[Zvar,Zval]
                                                         + alpha*filtdata[right_idx].loc[Zvar,Zval])
    print('    -> Finished interpolation.')

# Save chemtable file
print('\n')
print('Saving table as binary ...')
output_file_path = os.path.join(output_dir, output_filename)
ctt.write_chemtable_binary(output_file_path, finalData, '2ZTable')

print('\n')
print('Verifying table integrity ...')
testtable, testname = ctt.read_chemtable_binary(output_file_path, verbose=0)

if not testtable.equals(finalData):
    print("   -> FAILURE: Loaded table does not match saved, saving did not work properly")
    print("")
    print("Original Table: ")
    ctt.print_chemtable(finalData)
    print(finalData)
    print("")
    print("Saved table: ")
    ctt.print_chemtable(testtable)
    print(testtable)
    print("")
    raise RuntimeError("Saving table failed. quitting")
else:
    print("   -> SUCCESS. Saved table data:")
    print("")
    ctt.print_chemtable(finalData)
    print(finalData)
    ctt.print_chemtable(testtable)
    print(testtable)
