import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing    import StandardScaler
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import h5py
from sklearn.metrics          import mean_squared_error
from sklearn.metrics          import r2_score
import colors
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': "Times New Roman"})

##### FUNCTIONS #####

def compute_error_metrics(data, truedata, mse, filename, label):
    rmses = np.sqrt(mean_squared_error(truedata,data, multioutput='raw_values'))
    ranges = np.max(truedata) - np.min(truedata)
    r2s = r2_score(truedata,data, multioutput='raw_values')
    r2so= r2_score(truedata,data, multioutput='uniform_average')
    error_metrics = pd.DataFrame({'rmse':rmses,'range':ranges,'r2':r2s}, index = truedata.columns)
    error_metrics.loc['Overall'] = [np.sqrt(mse),0,r2so]
    error_metrics.to_csv(filename + '/'+ label + '.csv')
    return error_metrics

def convert_mol_mass(data):
    mws = {'POSF10325':154.29568,
           'CO2': 12.01 + 2*16.00,
           'CO' : 12.01 + 16.00,
           'O2' : 2 * 16.00,
           'OH' : 16.00 + 1.0079,
           'H2' : 2 * 1.0079,
           'CH4': 12.01 + 4*1.0079,
           'H2O': 16.00 + 2 * 1.0079}
    
    for col in data.columns:
        if col.startswith('RR'):
            specname = col.split('_')[-1]
            data[col] *= mws[specname]

def var2label(var, filtered=False):
    if var.startswith('prod_'):
        if not filtered:
            vartext = var.split(' ')[0].replace('prod_',"\\dot{\\omega}_{\\rm ")+'}'
        else:
            vartext = var.split(' ')[0].replace('prod_',"\\widetilde{\\dot{\\omega}}_{\\rm ")+'}'
        vartext = '$' + vartext + '$'
    elif var.startswith('RR_'):
        if not filtered:
            vartext = var.split(' ')[0].replace('RR_',"\\dot{\\omega}_{\\rm ")+'}'
        else:
            vartext = var.split(' ')[0].replace('RR_',"\\widetilde{\\dot{\\omega}}_{\\rm ")+'}'
        vartext = '$' + vartext + '$'
    elif var != var.split(' ')[0]:
        vartext = '$' + var.split(' ')[0] + '$' if not filtered else '$\\widetilde{' + var.split(' ')[0] + '}$'
    elif var == 'phi':
        vartext = r"$\phi$" if not filtered else r"$\bar{\phi}$"
    else:
        vartext = var
    if vartext == 'POSF10325':
        vartext = 'Fuel'

    vartext = var.replace(var.split(' ')[0], vartext)
    if  var.startswith('RR_'):
        vartext+= ' (kg/m3.s)'
    vartext = vartext.replace('/m3.s','/m$^3\cdot$s')
    return vartext

def load_and_scale_data(directory,
                        trainvars, passvars, predictvars,
                        scalers = {}, scalefunc = StandardScaler, with_variance = False,
                        data_source = 'npz'):
    # Load data
    if data_source == 'npz':
        loadfile = np.load(directory, allow_pickle=True)
        trndata  = pd.DataFrame(loadfile['trndata'], columns= loadfile['columns'])
        tstdata  = pd.DataFrame(loadfile['tstdata'], columns= loadfile['columns'])
    elif data_source == 's3d':
        with h5py.File(directory,'r') as dfile:
            trndata = pd.DataFrame({key.replace('?','/') : dfile['DNS'][key][-1,:,:].flatten()
            #trndata = pd.DataFrame({key.replace('?','/') : dfile['DNS'][key][:,:,-1].flatten()
                                    for key in dfile['DNS'].keys()
                                    if key != 'planes'})
            #for var in trndata.columns: print( var, np.min(trndata[var]), np.max(trndata[var]))
            tstdata = trndata.copy()
        
    elif data_source == 'flamelet':
        trndata = pd.read_csv(directory)
        tstdata = pd.read_csv(directory)
    else:
        raise RuntimeError("invalid data source specification")
    
    nsamples  = len(trndata.index)
    print('Training data set has {} samples'.format(nsamples))
    print('Testing  data set has {} samples'.format(len(tstdata.index)))

    # Split data into subsets
    trn_data = {}
    tst_data = {}
    trn_data['inp']  = trndata[trainvars].to_numpy()
    tst_data['inp']  = tstdata[trainvars].to_numpy()
    trn_data['pass'] = trndata[passvars].to_numpy()
    tst_data['pass'] = tstdata[passvars].to_numpy()
    trn_data['out']  = trndata[predictvars].to_numpy()
    tst_data['out']  = tstdata[predictvars].to_numpy()

    # Scale data - use existing scalers if they are input
    for dset in ['inp', 'pass', 'out']:
        if trn_data[dset].size > 0:
            try:
                trn_data[dset] = scalers[dset].transform(trn_data[dset])
            except KeyError:
                scalers[dset] = scalefunc()
                trn_data[dset] = scalers[dset].fit_transform(trn_data[dset])
            tst_data[dset] = scalers[dset].transform(tst_data[dset])

    # Add variances if required
    if with_variance:
        trn_data['var'] = extract_variances(trndata, trainvars, scalers['inp'])
        tst_data['var'] = extract_variances(tstdata, trainvars, scalers['inp'])

    if data_source == 'flamelet':
        return nsamples, trn_data, trndata['z (m)'], scalers
    else:
        return nsamples, trn_data, tst_data, scalers
        
def extract_variances(dataframe, trainvars, scaler):
    # Set up scaling information
    scales = dict(zip(trainvars,1.0/scaler.scale_))
    try:
        offsets = dict(zip(trainvars, -scaler.mean_/scaler.scale_))
    except:
        offsets = dict(zip(trainvars, np.zeros(len(trainvars))))
        
    # Extract moments, verifying that all exist
    nmoments = len(trainvars) **2
    variances = np.zeros((len(dataframe.index), nmoments))
    moments = [combo for combo in itertools.product(trainvars,repeat=2)]
    for idex, moment in enumerate(moments):
        if "~".join(moment) in dataframe.columns:
            variances[:,idex] = dataframe["~".join(moment)]
        elif "~".join(reversed(moment)) in dataframe.columns:
            variances[:,idex] = dataframe["~".join(reversed(moment))]
        else:
            raise RuntimeError("Moment " + "~".join(moment) + " not found in DataFrame")
        
        variances[:,idex] = (scales[moment[0]]*scales[moment[1]]*variances[:,idex] +
                             scales[moment[0]]*offsets[moment[1]]*dataframe[moment[0]] +
                             scales[moment[1]]*offsets[moment[0]]*dataframe[moment[1]] +
                             offsets[moment[0]]*offsets[moment[1]])
    return variances

# Class to contain various bits of metadata
class metadata:

    def __init__(self,directory):
        self.directory = directory

    def load(self):
        self.__dict__ = np.load(self.directory + '/metadata.npz',
                                allow_pickle=True)['dictionary'][0]

    def save(self):
        np.savez(self.directory + '/metadata.npz',
                 dictionary = [self.__dict__,])
        
    def __repr__(self):
        print('\n Metadata container with the following variables:\n')
        for key,val in self.__dict__.items():
            print(key, ' : ')
            print(val)
            print('')
        return ('End container')
        
    def save_net_info(self, fname, varname_converter=None):
        """Save to metadata file readable by PelePhysics."""
        
        def insert_line_breaks(strs, breakpoint=100, prefix_len=11):
            
            nchars = prefix_len
            ins_points = []
            
            for i in range(len(strs)):
                nchars += len(strs[i]) + 1
                if nchars >= breakpoint-1:
                    ins_points.append(i)
                    nchars = prefix_len + len(strs[i]) + 1
                    
            for i in range(len(ins_points)):
                strs.insert(ins_points[i]+i, '\\\n'+' '*(prefix_len-1))
        
        with open(fname, 'w') as file:
            
            print("# Name of the neural network model", file=file)
            print(f"model_name = {self.model_name}", file=file)
            ndim = len(self.passvars) + self.nmanivars
            print("# Number of input dimensions", file=file)
            print(f"ndim = {ndim}", file=file)
            nvar = len(self.predictvars)
            print("# Number of output dimensions", file=file)
            print(f"nvar = {nvar}", file=file)
            print("# Number of manifold parameters", file=file)
            print(f"nmanpar = {self.nmanivars}", file=file)
            
            dimnames = list(map(varname_converter, self.passvars))
            dimnames += [f'xi{i}' for i in range(self.nmanivars)]
            dimnamelist = dimnames.copy()
            insert_line_breaks(dimnames)
            dimnames = ' '.join(dimnames)
            print("# Names of input variables", file=file)
            print(f"dimnames = {dimnames}", file=file)
            
            varnames = list(map(varname_converter, self.predictvars))
            insert_line_breaks(varnames)
            varnames = ' '.join(varnames)
            print("# Names of output variables", file=file)
            print(f"varnames = {varnames}", file=file)
            
            print("# Definitions of input variables", file=file)
            
            for i in range(len(self.passvars)):
                
                pardef = dimnamelist[i]
                print(f"def_{dimnamelist[i]} = {pardef}", file=file)
                
            for i in range(len(self.passvars), len(dimnamelist)):
                
                weights = map(lambda w: str(w.item()), self.xidefs[i-len(self.passvars)])
                pardef = zip(weights, '*'*len(self.trainvars), map(varname_converter, self.trainvars))
                pardef = ["".join(x) for x in pardef]
                lhs = f"def_{dimnamelist[i]}"
                insert_line_breaks(pardef, prefix_len=len(lhs)+3)
                pardef = ' '.join(pardef)
                print(f"{lhs} = {pardef}", file=file)
                
            print("# Biases to be used calculating input variables", file=file)
            manibiases = ' '.join(map(lambda mb: str(mb.item()), self.manibiases))
            print(f"manibiases = {manibiases}", file=file)

# handle pytorch data
def vh(data, dev=torch.device("cpu"), dtype=torch.float):
    return torch.as_tensor(data,dtype=dtype).to(device=dev)
def vh1(data, dev=torch.device("cpu"), dtype=torch.float):
    return  torch.autograd.Variable(torch.as_tensor(data,dtype=dtype).to(device=dev))

#### Plots ####

def GetVarLims(var, lims=None, data=None, expandrange=True):
    if lims is None  and data is None:
        raise RuntimeError("GetVarLims: must specify lims or data")

    get_from_data = True
    
    if lims is not None:
        if var in lims.keys():
            minx, maxx = lims[var]
            get_from_data=False

    if get_from_data:
        minx = np.min(data[var])
        maxx = np.max(data[var])
        if expandrange:
            rwidth = maxx-minx
            center = (maxx + minx) /2
            minx = center - 0.6*rwidth 
            maxx = center + 0.6*rwidth

    return minx, maxx

# bar plot of component weights
def MakeManiDefPlots(outfile, md, model, lineage='', rescale=False, resign=False, with_legend=False, log=False):

    print('Making Mani Def plots')
    
    labels = md.trainvars
    x = np.arange(len(labels))
    width = 0.7/md.nmanivars 
    offset = -0.5*(md.nmanivars - 1.0)

    figheight = 2 if not log else 2.5
    fig,ax = plt.subplots(figsize=(colors.twocol_figsize[0], figheight))

    scaleweights = model.inputs['manidef'].cpu().numpy().copy()
    if rescale : scaleweights /= md.scalers['inp'].scale_[...,None] 
    scaleweights /= np.max(np.abs(scaleweights),0)[None,...]
    # Below: multiply by -1 if max abs weight is negative
    if resign : scaleweights *= np.diag(scaleweights[np.argmax(np.abs(scaleweights),0)] )[None,...]  

    if log: scaleweights = np.abs(scaleweights)

    for manivar in range(md.nmanivars):
        ax.bar(x + (manivar + offset)*width, scaleweights[:,manivar],
               width, label=r'$\xi_{}$'.format(manivar+1), color=colors.get_color(index=manivar))

    ax.set_xticks(x)
    ax.tick_params('x', which='major', length=0)
    ax.tick_params('x', which='minor', length=5, width=1, direction = 'inout')
    ax.set_xticks(np.arange(len(x)+1)-0.5, minor=True)
    ax.set_xticklabels([var2label(label) for label in labels],rotation=45)
    ax.xaxis.label.set_size(14)
    ax.axhline(color='black')
    if with_legend: ax.legend(bbox_to_anchor=(1.04,1), loc='upper left')
    if log:
        plt.ylim(1e-6,2)
        plt.yscale('log')
        ax.set_ylabel(r'$|W_{ij}|$')
    else:
        plt.ylim(-1.05,1.05)
        ax.set_ylabel(r'$W_{ij}$')
    print(colors.twocol_figbounds)
    ax.set_position([colors.twocol_figbounds[0],0.25, colors.twocol_figbounds[2],0.73])

    legtext = ''
    if with_legend: legtext = '_leg'
    plt.savefig(outfile + '/manidef'+lineage+legtext+'.png', dpi=150)
    if not with_legend:
        plt.figure(figsize=(1.0,figheight))
        plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',borderpad=0.03, handlelength = 0.6, frameon=False)
        plt.savefig(outfile + '/manidef'+lineage+'_leg.png', dpi=150)
        plt.clf()
        plt.close()
    plt.clf()
    plt.close()
    
# scatter plot in manifoldspace
def MakeManiPlots(outfile, manivars, outdata=None, manivars2=None, outdata2=None, 
                  colorvars=['T (K)'], label='true', lineage='', nplot=20000, with_cb = False,
                  varlims=None, modify_ticks=True, filtered=False, c1='red', c2='black'):
    print('Making Mani Plots')
    
    nsamples = manivars.shape[0]
    if nplot < nsamples:
        chosen = np.random.choice(nsamples,nplot)
    else:
        chosen = np.arange(nsamples)
    if manivars2 is not None:
        nsamples2 = manivars2.shape[0]
        if nplot < nsamples:
            chosen2 = np.random.choice(nsamples2,nplot)
        else:
            chosen2 = np.arange(nsamples2)
        
    if lineage is not '' : lineage = '_' + lineage
    for var in colorvars:
        plt.figure(figsize=(colors.twocol_figsize))
        if outdata is not None:
            #FIXME : remove lognorm
            minc,maxc = GetVarLims(var, varlims, outdata, expandrange=False)
            print(minc,maxc)
            clp = plt.scatter(manivars[chosen,0],manivars[chosen,1],c=outdata[var][chosen],
                        #norm=LogNorm(vmin=minc, vmax=maxc),s=0.4)
                        vmin=minc, vmax=maxc,s=0.4)
            plt.clim([minc,maxc])

        else : 
            plt.scatter(manivars[chosen,0],manivars[chosen,1],color=c1,s=0.4)
                
        if manivars2 is not None:
            if outdata2 is not None:
                plt.scatter(manivars2[chosen2,0],manivars2[chosen2,1],c=outdata2[var][chosen2],s=0.4)
                plt.clim([minc,maxc])
            else:
                plt.scatter(manivars2[chosen2,0],manivars2[chosen2,1],color=c2,s=0.4)

        if not filtered :
            plt.xlabel(r'$\xi_1$')
            plt.ylabel(r'$\xi_2$')
        else:
            plt.xlabel(r'$\widetilde{\xi}_1$')
            plt.ylabel(r'$\widetilde{\xi}_2$')
            
        if modify_ticks:
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))

        minlim = 2.1
        lim = plt.gca().get_xlim()
        plt.xlim([lim[0] if lim[0] < -minlim else -minlim, lim[1] if lim[1] > minlim else minlim])
        lim = plt.gca().get_ylim()
        plt.ylim([lim[0] if lim[0] < -minlim else -minlim, lim[1] if lim[1] > minlim else minlim])
        #plt.gcf().tight_layout()
        
        plt.gca().set_position(colors.twocol_figbounds)
        plt.savefig(outfile + '/mani_' + label + '_' + var.split(' ')[0]+ lineage+'.png', dpi=150)
        if with_cb:
            fig = plt.figure(figsize = (1.0, colors.twocol_figsize[1]))
            figbounds = (0.01,colors.twocol_figbounds[1], 0.18, colors.twocol_figbounds[3])
            ax = fig.add_axes(figbounds)
            cb  = plt.colorbar(clp, cax=ax)
            cb.set_label(var2label(var, filtered=filtered))
            plt.savefig(outfile + '/mani_' + label + '_' + var.split(' ')[0]+ lineage+'_cb.png', dpi=150)
            plt.clf()
            plt.close()
        plt.clf()
        plt.close()

def MakeParityPlotsVec(outfile, true_values_vec, pred_values_vec, 
                       parityvars=['T (K)'], filtered=False,
                       lineage='', nplot=20000, varlims=None,
                       colors_vec = ['r','k'], labels_vec=['Training','Validation'], val_labels=['Validation']):

    print('Making parity plots')
    if lineage is not '' : lineage = '_' + lineage

    for var in parityvars:
        plt.figure(var, figsize = colors.twocol_figsize)
    
    for ii in range(len(true_values_vec)):
        nsamples = true_values_vec[ii].shape[0]
        if nplot < nsamples:
            chosen = np.random.choice(nsamples,nplot)
        else:
            chosen = np.arange(nsamples)

        lab = ' - ' + labels_vec[ii] if labels_vec[ii] != '' else ''
            
        for var in parityvars:
            plt.figure(var)
            r2 = r2_score(true_values_vec[ii][var], pred_values_vec[ii][var])
            plt.scatter(true_values_vec[ii][var][chosen],pred_values_vec[ii][var][chosen],color=colors_vec[ii],s=0.4,
                        label=r"$r^2 = {:6.4f}$".format(r2)+lab)

    for var in parityvars:
        plt.figure(var)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        handles_val = [handle for handle, label in zip(handles, labels) if any([val_label in label for val_label in val_labels])]
        labels_val = [label for label in labels if any([val_label in label for val_label in val_labels])]
        handles_trn = [handle for handle in handles if handle not in handles_val]
        labels_trn = [label for label in labels if label not in labels_val]

        leg1 = plt.legend(handles_val, labels_val, frameon=False, loc='upper left', title= 'Validation',
                   scatterpoints=4, handlelength=0.9, handletextpad=0.2,
                   labelspacing=0.1,  fontsize='small', borderpad=0.05)
        
        leg2 = plt.legend(handles_trn, labels_trn, frameon=False, loc='lower right', title='Training',
                   scatterpoints=4, handlelength=0.9, handletextpad=0.2,
                   labelspacing=0.1,  fontsize='small', borderpad=0.05)
        plt.gca().add_artist(leg1)
        
        minx, maxx = GetVarLims(var, lims=varlims, data=true_values_vec[0])
        plt.xlim([minx,maxx])
        plt.ylim([minx,maxx])
        plt.plot([minx,maxx],[minx,maxx],'k:')
        vartext = var2label(var, filtered=filtered)
        plt.xlabel(r"{} - True".format(vartext))
        plt.ylabel(r"{} - Pred.".format(vartext))
        plt.gca().set_position(colors.twocol_figbounds)
        plt.savefig(outfile + '/parity_' + var.split(' ')[0] + lineage + '.png', dpi=150)
        plt.clf()
        plt.close()
            
# scatter parity plot
def MakeParityPlots(outfile,
                    true_out_tst, pred_out_tst,
                    true_out_trn=None, pred_out_trn=None,
                    parityvars=['T (K)'],
                    lineage='', nplot=20000,varlims=None,
                    lab2='Training', lab1='Validation',
                    c2='k',c1='r', filtered=False):
    print('Making parity plots')
    
    if true_out_trn is not None:
        nsamples_trn = true_out_trn.shape[0]
        if nplot < nsamples_trn:
            chosen_trn = np.random.choice(nsamples_trn,nplot)
        else:
            chosen_trn = np.arange(nsamples_trn)
    
    nsamples_tst = true_out_tst.shape[0]
    if nplot < nsamples_tst:
        chosen_tst = np.random.choice(nsamples_tst,nplot)
    else:
        chosen_tst = np.arange(nsamples_tst)
    
    if lineage is not '' : lineage = '_' + lineage
    for var in parityvars:
        plt.figure(figsize=(colors.twocol_figsize))
        minx, maxx = GetVarLims(var, lims=varlims, data=true_out_tst)
        plt.xlim([minx,maxx])
        plt.ylim([minx,maxx])
        if true_out_trn is not None:
            r2_trn = r2_score(true_out_trn[var], pred_out_trn[var])
            plt.scatter(true_out_trn[var][chosen_trn],pred_out_trn[var][chosen_trn],c=c2,s=0.4,
                        label=r"$r^2 = {:6.4f}$ - ".format(r2_trn)+lab2)
            #plt.text(0.05,0.9,r"$r^2 = {:6.4f}$ - ".format(r2_trn)+lab2, transform = plt.gca().transAxes, c=c2)
        r2_tst = r2_score(true_out_tst[var], pred_out_tst[var])
        plt.scatter(true_out_tst[var][chosen_tst],pred_out_tst[var][chosen_tst],c=c1,s=0.4,
                    label=r"$r^2 = {:6.4f}$ - ".format(r2_tst)+lab1)
        #plt.text(0.05,0.8,r"$r^2 = {:6.4f}$ - ".format(r2_tst)+lab1, transform = plt.gca().transAxes, c=c1)
        plt.legend(frameon=False, loc='upper left',
                   scatterpoints=4, handlelength=0.9, handletextpad=0.2,
                   labelspacing=0.1,  fontsize='medium', borderpad=0.2)
        plt.plot([minx,maxx],[minx,maxx],'k:')
        vartext = var2label(var, filtered=filtered)
        plt.xlabel(r"{} - True".format(vartext))
        plt.ylabel(r"{} - Pred.".format(vartext))
        plt.gca().set_position(colors.twocol_figbounds)
        plt.savefig(outfile + '/parity_' + var.split(' ')[0] + lineage + '.png', dpi=150)
        plt.clf()
        plt.close()

# Plot a slice of DNS data
def MakeSlicePlots(outfile, data, slicevars, nmanivars=0, nvars=0, size=(832,832), name='',
                   lab='',varlims=None, with_cb=False, crop=0, figsize=(4,4), scalebar=None, filtered=False):
    print('Making slice plots')
    
    # Attach column names if plotting manifold variables
    if nmanivars > 0:
        slicevars = ['xi_'+str(manivar+1) for manivar in range(nmanivars)]
        if nvars > 0:
            slicevars += ['xi_var_'+str(ivar+1) for ivar in range(nvars)]
        data = pd.DataFrame(data[:,:nmanivars+nvars].numpy(),columns=slicevars)
    
    #name = name.split('/')[-1].split('_')[0] + '_'
    name = name.split('/')[-1].replace('.h5','_')

    for var in slicevars:
        plt.figure(figsize=figsize)
        if crop==0:
            cropy = 0; cropx =0;
            res = plt.imshow(data[var].to_numpy().reshape(size)[:,:])
        elif isinstance(crop,int):
            cropx = crop; cropy = crop 
            plotdata = data[var].to_numpy().reshape(size)[crop:-crop,crop:-crop]
            print('cropping to ', plotdata.shape)
            res = plt.imshow(plotdata)
        else:
            cropx = crop[1]; cropy = crop[0]
            plotdata = data[var].to_numpy().reshape(size)[crop[0]:-crop[0],crop[1]:-crop[1]]
            print('cropping to ', plotdata.shape)
            res = plt.imshow(plotdata)
            #res.cmap.set_under('w')
        #res.cmap.set_over('w')

        vartext = var2label(var, filtered=filtered)
        #if with_cb: plt.colorbar(orientation='vertical').set_label(vartext)
        plt.gca().tick_params(direction='inout',
                              top=False,right=False, left=False, bottom=False,
                              labelleft=False,labelbottom=False)
        minx, maxx = GetVarLims(var, lims=varlims, data=data)
        plt.clim(minx,maxx)
        if scalebar is not None and var=='T (K)':
            print('making scalebar')
            scalelen = 1000 / scalebar
            ncellsy = size[0] - 2*cropy
            ncellsx = size[0] - 2*cropx
            height = 0.07*ncellsy
            start = 0.1*ncellsx
            plt.plot([start+scalelen,start],[height,height],'w-',lw=5)
            plt.text(start+scalelen/2,height+0.04*ncellsy, '1 mm',
                     horizontalalignment='center', verticalalignment='top',c='white')
        plt.tight_layout(pad=0.32)
        pos = plt.gca().get_position()
        shift = colors.twocol_figbounds[1] + colors.twocol_figbounds[3] - pos.y1
        pos.y1 += shift; pos.y0 +=shift
        plt.gca().set_position(pos)

        if with_cb:
            cbfig = plt.figure(figsize=[1,figsize[1]])
            figbounds = (0.01,pos.y0, 0.18, pos.y1-pos.y0)
            ax = cbfig.add_axes(figbounds)
            cb  = plt.colorbar(res, cax=ax)
            cb.set_label(var2label(var, filtered=filtered))
            plt.savefig(outfile +'/slice_'+ name + var.split(' ')[0] + '_' + lab + '_cb.png', dpi=150)
            plt.clf()
            plt.close()
            
            
        plt.savefig(outfile +'/slice_'+ name + var.split(' ')[0] + '_' + lab +'.png', dpi=150)
        plt.clf()
        plt.close()


    
                   
if __name__ == "__main__":
    import sys
    fname = sys.argv[1]
    md = metadata(fname)
    md.load()
    print(md)
