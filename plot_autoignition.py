import matplotlib.pyplot as plt
import net_helper as nh
import manifold_reduction_nets as mrn
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
import colors
matplotlib.use('Agg')
plt.rcParams.update({'text.usetex': False})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': "Times New Roman"})

# Inputs
# load network
#case = 'l1'
savepath = 'saved_networks/autoignition'

for subdir in ['raw-pca-2', 'raw-cmlm-2', 'raw-fgm-2']:
    #for subdir in ['raw-cpt-2']:

    useGPU = False
    plotvars = np.array(['SRC_CO2','SRC_H2O','T',"Y-CO"])
    colorvars = plotvars
    plotpath = savepath.replace("saved_networks","plots")

    varlims = { 'OH' : [-0.001, 0.006],
                'CO2' : [-0.01, 0.18],
                'SRC_CO2' : [-150e-3,1200e-3],
                'SRC_H2O' : [-150e-3,1200e-3],
                'T' : [1200, 3000]
                }

    # Set up device
    if useGPU:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Load metadata
    md = nh.metadata(os.path.join(savepath))
    md.load()

    dir1 = md.directory_raw

    print('Loading data from :\n{}'.format(dir1))
    nsamples_rawf, trn_rawf, tst_rawf, scalers = nh.load_and_scale_data(dir1,
                                                                        md.trainvars, md.passvars, md.predictvars,
                                                                        scalers=md.scalers,
                                                                        with_variance=False)

    md = nh.metadata(os.path.join(savepath))
    md.load()
    with open(os.path.join(savepath, subdir, 'best_trial.txt')) as fi:
        trial = fi.read()
    
    model, optimizer = mrn.load_checkpoint(filename = os.path.join(savepath, subdir, trial+'.npz'),
                                           generate_new=True, verbose=False)
    if model.net_type is 'ManifoldReductionNet': model = mrn.manifold2prediction(model)
    model = mrn.unscale_prediction_net(model, md.scalers)
    model.eval()
    
    # exercise network
    def get_invars(dataset,model):
        return model.calc_manifold(nh.vh(np.concatenate((scalers['inp'].inverse_transform(dataset['inp']),
                                                         dataset['pass']),1)))
    
    filename = os.path.join(plotpath,subdir)
    if not os.path.exists(filename): os.makedirs(filename)
    nh.MakeManiDefPlots(filename, md, model)
    
    invars = get_invars(tst_rawf,model)
    outpred = (pd.DataFrame(model(invars).detach().cpu().numpy(), columns=md.predictvars))
    outtrue = (pd.DataFrame(md.scalers['out'].inverse_transform(tst_rawf['out']                     ), columns=md.predictvars))
    invars_trn = get_invars(trn_rawf,model)
    outpred_trn = (pd.DataFrame(model(invars_trn).detach().cpu().numpy(), columns=md.predictvars))
    outtrue_trn = (pd.DataFrame(md.scalers['out'].inverse_transform(trn_rawf['out']                     ), columns=md.predictvars))

    nh.MakeParityPlots(filename, outtrue, outpred, outtrue_trn, outpred_trn,
                       parityvars=plotvars, varlims=varlims,
                       c1=colors.get_color(color='blue',shade='d') ,
                       c2=colors.get_color(color='blue',shade='l') )
    
    nh.MakeManiPlots(filename, invars_trn[:,:md.nmanivars], outtrue_trn,
                     label='true', colorvars=['T'], with_cb=True, modify_ticks=True)
        


