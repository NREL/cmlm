# import libraries
import pandas as pd
import numpy as np
import os
import torch
from torch import nn, optim, device, randperm
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import manifold_reduction_nets as mrn
import net_helper as nh

# reproducability
torch.manual_seed(31)
np.random.seed(31)

# options
useGPU = False

# directory where everything will be saved
savepath = 'saved_networks/flamesheet/'
if not os.path.exists(savepath): os.makedirs(savepath)

# Create and fill container for various options
md = nh.metadata(savepath)
md.model_name = 'flamesheet'
md.directory_raw  = 'data/test_flamesheet_data.npz' # data directory
md.network = (50,50) # network structure
md.loss_alpha  = 0.01 # regularization parameter
md.nmanivars = 1 # dimensionality of manifold
md.save_chk = False # save network at every epoch when training
md.fuel = 'CH4' # Fuel species 

# Whether or not to use SHERPA population-based training for hyperparameter optimization
md.use_sherpa = False                                               
# Net training options - for SHERPA population based (expensive)
if md.use_sherpa:
    md.batchsize = [8, 16, 32]
    md.learningrate = [1e-6,1e-02]
    md.nepochs = 10
    md.ngens = 15
    md.nsibs = 8
    md.reduce_lr = 0
    md.stop_no_progress = 0
# Net training options - for plateau learning rate schedule (cheap)
else:
    md.batchsize =[16]
    md.learningrate = [1e-02,1e-02]
    md.nepochs = 400
    md.ngens = 1
    md.nsibs = 1 ## FIXME
    md.reduce_lr = 3
    md.stop_no_progress = 10
    
# Variables that the manifold variables may be linear combinations of
md.trainvars = np.array(['Y-CO2','Y-H2','Y-N2','Y-CO','Y-O2','Y-H2O','Y-'+md.fuel,'Y-OH'])

# Variables passed through directly as inputs to the prediction nets
md.passvars = np.array([])

# Variables that get predicted by the prediction net
md.predictvars = list(md.trainvars) + ['SRC_' + v[2:] for v in md.trainvars if (v != 'Y-N2')] + \
        ['T', 'RHO', 'DIFF', 'VISC']
md.predictvars = np.array(md.predictvars)

# Print all inputs and save to file
print('**** Inputs for training networks ****\n', md, '\n\n')
md.save()

# Definition of FGM manifold (Progress variable)
xidefs_flt = [['Y-CO','Y-CO2','Y-H2O','Y-H2']]

##### RUN ######

# Set loss functions
lossfunc = nn.MSELoss()
netlossfunc = mrn.netstruct_loss('l1l2',lam1=md.loss_alpha/md.nmanivars, ramp=0).calc_loss

# set function to scale data
def scalefunc(): return StandardScaler(with_mean = True)

# Apply GPU settings
if useGPU:
    dev = device("cuda")
else:
    dev = device("cpu")

# Initialize some quantities
md.trnloss ={}; md.tstloss = {}
npassvars = len(md.passvars)
nout      = len(md.predictvars)

# Load Data
nsamples_raw, trn_raw, tst_raw, scalers = nh.load_and_scale_data(md.directory_raw,
                                                                 md.trainvars, md.passvars, md.predictvars,
                                                                 scalefunc = scalefunc)

# Save data scalers with metadata
md.scalers = scalers
md.save()

#--- Train FGM+ANN model ---#
label = 'raw-fgm-'+ str(md.nmanivars)

# Get weights matrix W from specified manifold variables
xidefs_flt = np.array( [[1.0 if var in xidef else 0.0 for var in md.trainvars] for xidef in xidefs_flt] ).T
xidefs_flt *= scalers['inp'].scale_[...,None] # account for scaling of data
xidefs_flt *= 1.0/np.sqrt(np.sum(xidefs_flt**2,0))[None,...] # normalize magnitudes
xidefs_flt[:,-1] *= -1.0
nmv = min(md.nmanivars,xidefs_flt.shape[1])

# Initialize network and function to extract network inputs from loaded data structures
flt_net = mrn.PredictionNet(nmv+npassvars,md.network,nout,manidef=xidefs_flt[:,:nmv])
def get_invars_flt(dataset):
    return flt_net.calc_manifold(torch.as_tensor(np.concatenate((dataset['inp'], dataset['pass']),1),dtype=torch.float))

# Train network
md.trnloss[label], md.tstloss[label] = mrn.train_net_sherpa(get_invars_flt(trn_raw),trn_raw['out'],
                                                            get_invars_flt(tst_raw),tst_raw['out'],
                                                            flt_net, lossfunc=lossfunc, dev=dev,
                                                            nepochs=md.nepochs, ngenerations=md.ngens, nsiblings=md.nsibs,
                                                            batchsize=md.batchsize, learning_rate=md.learningrate,
                                                            reduce_lr=md.reduce_lr, stop_no_progress=md.stop_no_progress,
                                                            savepath=savepath + '/'+label,
                                                            plot_dashboard='dashboard')

#--- Train PCA+ANN model ---#
label = 'raw-pca-'+ str(md.nmanivars)

# run PCA to define manifold
pca = PCA(n_components = md.nmanivars)
components = pca.fit_transform(trn_raw['inp'])
xidefs_pca = pca.components_.T

# Initialize network and function to extract network inputs frokm loaded data structures
pca_net = mrn.PredictionNet(md.nmanivars+npassvars,md.network,nout,manidef=xidefs_pca)
def get_invars_pca(dataset):
    return pca_net.calc_manifold(torch.as_tensor(np.concatenate((dataset['inp'], dataset['pass']),1),dtype=torch.float))

# Train Network
md.trnloss[label], md.tstloss[label] = mrn.train_net_sherpa(get_invars_pca(trn_raw),trn_raw['out'],
                                                            get_invars_pca(tst_raw),tst_raw['out'],
                                                            pca_net, lossfunc=lossfunc, dev=dev,
                                                            nepochs=md.nepochs, ngenerations=md.ngens, nsiblings=md.nsibs,
                                                            batchsize=md.batchsize, learning_rate=md.learningrate,
                                                            reduce_lr=md.reduce_lr, stop_no_progress=md.stop_no_progress,
                                                            savepath=savepath + '/'+label,
                                                            plot_dashboard='dashboard')

#--- Train Co-optimized ML Manifolds  model ---#
label = 'raw-cmlm-'+ str(md.nmanivars)

# Initialize network (using PCA definition) and function to extract network inputs frokm loaded data structures
cpt_net = mrn.get_manifold_net(flt_net, reinit=True, Npass=npassvars)
def get_invars_cpt(dataset):
    return np.concatenate((dataset['inp'], dataset['pass']),1)

# Train network
md.trnloss[label], md.tstloss[label] = mrn.train_net_sherpa(get_invars_cpt(trn_raw),trn_raw['out'],
                                                            get_invars_cpt(tst_raw),tst_raw['out'],
                                                            cpt_net, lossfunc=lossfunc, dev=dev, wgtlossfunc=netlossfunc,
                                                            nepochs=md.nepochs, ngenerations=md.ngens, nsiblings=md.nsibs,
                                                            batchsize=md.batchsize, learning_rate=md.learningrate,
                                                            reduce_lr=md.reduce_lr, stop_no_progress=md.stop_no_progress,
                                                            savepath=savepath + '/'+label,
                                                            plot_dashboard='dashboard', save_chk = md.save_chk)

# Save and Print final results
md.save()
xidefs_cpt = cpt_net.state_dict()['manifold.weight'].cpu().numpy().T
xidefs = {'FGM':xidefs_flt, 'PCA':xidefs_pca, 'CMLM':xidefs_cpt}

print('\n****Manifold definitions for each method****\n')
for label in ['FGM','PCA','CMLM']:
    print(label)
    print(pd.DataFrame(xidefs[label], columns=np.arange(md.nmanivars), index=md.trainvars))
    print('')

print('****Overall MSEs****\n')
finaldata = pd.DataFrame({'Training':md.trnloss, 'Testing':md.tstloss})
print(finaldata)

# Save net
cpt_net.to(device("cpu"))
pred_net = mrn.manifold2prediction(cpt_net)
pred_net = mrn.unscale_prediction_net(pred_net, md.scalers)
torch.jit.script(pred_net).save("net.pt")

# Save metadata in PelePhysics-readable format
def munge_varname(s):
    s = s.split()[0].upper()
    return s
    
md.xidefs = pred_net.inputs['manidef'].T
md.manibiases = pred_net.inputs['manibiases']
md.save_net_info("net_info.txt", varname_converter=munge_varname)    
