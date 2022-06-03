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
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': "Times New Roman"})

# Inputs
# load network
#case = 'l1'
savepath = 'saved_networks/strained'

for subdir in ['raw-pca-2', 'raw-cmlm-2', 'raw-fgm-2']:
    #for subdir in ['raw-cpt-2']:

    useGPU = False
    plotvars = np.array(['RR_CO2','RR_H2O','T (K)',"CO"])
    colorvars = plotvars
    plotpath = savepath.replace("saved_networks","plots")

    varlims = { 'OH' : [-0.001, 0.006],
                'CO2' : [-0.01, 0.18],
                'RR_CO2' : [-20,280],
                'RR_H2O' : [-30,280],
                'T (K)' : [200, 2700]
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
    dir2 = 'data/strained_flames_phi1_allTP_kmeans_100000.npz'

    print('Loading data from :\n{}'.format(dir1))
    nsamples_rawf, trn_rawf, tst_rawf, scalers = nh.load_and_scale_data(dir1,
                                                                        md.trainvars, md.passvars, md.predictvars,
                                                                        scalers=md.scalers,
                                                                        with_variance=False)
    print('Loading data from :\n{}'.format(dir2))
    nsamples_nom, trn_nom, tst_nom, scalers = nh.load_and_scale_data(dir2,
                                                                     md.trainvars, md.passvars, md.predictvars,
                                                                     scalers=md.scalers,
                                                                     with_variance=False)
    outdata = {}
    md = nh.metadata(os.path.join(savepath))
    md.load()
    trial_results = pd.read_csv(os.path.join(savepath, subdir, 'results.csv'))
    trial_results = trial_results[trial_results['Status']=='COMPLETED']
    trial_results = trial_results.sort_values(by='Objective')
    ntrials = trial_results.shape[0]
    trials = [str(trial) for trial in trial_results['Trial-ID'] ]
    val_loss = []
    trn_loss = []
    for trial in trials[0:1]:
        print(trial)
        model, optimizer = mrn.load_checkpoint(filename = os.path.join(savepath, subdir, trial+'.npz'),
                                               generate_new=True, verbose=False)
        if model.net_type is 'ManifoldReductionNet': model = mrn.manifold2prediction(model)
        model.eval()

        # exercise network
        def get_invars(dataset,model):
            return model.calc_manifold(nh.vh(np.concatenate((dataset['inp'], dataset['pass']),1)))
        hist = pd.read_csv(os.path.join(savepath, subdir, trial+'.csv'))
        best_epoch = hist['val_loss'].idxmin()
        val_loss.append(hist['val_loss'][best_epoch])
        trn_loss.append(hist['trn_loss'][best_epoch])
        order = len(val_loss)

        filename = os.path.join(plotpath,subdir)
        if not os.path.exists(filename): os.makedirs(filename)
        if model.inputs['manidef'][4,0] > 0:
            model.inputs['manidef'][:,0] *= -1
            flip =True
        else:
            flip = False
        model.inputs['manidef'][:,1] *= -1
        nh.MakeManiDefPlots(filename, md, model, lineage='t'+trial+'_r'+str(order))
        if flip:
            model.inputs['manidef'][:,0] *= -1
        model.inputs['manidef'][:,1] *= -1
        print(trial, val_loss[-1], trn_loss[-1])

        if order in [1]:
            invars = get_invars(tst_rawf,model)
            outpred = (pd.DataFrame(md.scalers['out'].inverse_transform(model(invars).detach().cpu().numpy()), columns=md.predictvars))
            outtrue = (pd.DataFrame(md.scalers['out'].inverse_transform(tst_rawf['out']                     ), columns=md.predictvars))
            invars_trn = get_invars(trn_rawf,model)
            outpred_trn = (pd.DataFrame(md.scalers['out'].inverse_transform(model(invars_trn).detach().cpu().numpy()), columns=md.predictvars))
            outtrue_trn = (pd.DataFrame(md.scalers['out'].inverse_transform(trn_rawf['out']                     ), columns=md.predictvars))

            if 'cmlm' in subdir:
                print(md.trainvars)
                print(np.min(tst_rawf['inp'],axis=0))
                print(np.max(tst_rawf['inp'],axis=0))
                print(np.min(invars.detach().cpu().numpy(),axis=0))
                print(np.max(invars.detach().cpu().numpy(),axis=0))
                print(model.inputs['manidef'])
            
            nh.convert_mol_mass(outpred)
            nh.convert_mol_mass(outtrue)
            nh.convert_mol_mass(outpred_trn)
            nh.convert_mol_mass(outtrue_trn)

            nh.compute_error_metrics(outpred, outtrue, val_loss[-1], filename, 'error_tst_'+str(order))
            nh.compute_error_metrics(outpred_trn, outtrue_trn, trn_loss[-1], filename, 'error_trn_'+str(order))

            nh.MakeParityPlots(filename, outtrue, outpred, outtrue_trn, outpred_trn,
                               parityvars=plotvars, varlims=varlims, lineage=str(order),
                               c1=colors.get_color(color='blue',shade='d') ,
                               c2=colors.get_color(color='blue',shade='l') )
            
            invars = get_invars(tst_nom,model)
            outpred = (pd.DataFrame(md.scalers['out'].inverse_transform(model(invars).detach().cpu().numpy()), columns=md.predictvars))
            outtrue = (pd.DataFrame(md.scalers['out'].inverse_transform(tst_nom['out']                     ), columns=md.predictvars))
            invars_trn = get_invars(trn_nom,model)
            outpred_trn = (pd.DataFrame(md.scalers['out'].inverse_transform(model(invars_trn).detach().cpu().numpy()), columns=md.predictvars))
            outtrue_trn = (pd.DataFrame(md.scalers['out'].inverse_transform(trn_nom['out']                     ), columns=md.predictvars))

            if 'cmlm' in subdir:
                print(md.trainvars)
                print(np.min(tst_nom['inp'],axis=0))
                print(np.max(tst_nom['inp'],axis=0))
                print(np.min(invars.detach().cpu().numpy(),axis=0))
                print(np.max(invars.detach().cpu().numpy(),axis=0))
                print(model.inputs['manidef'])
            
            nh.convert_mol_mass(outpred)
            nh.convert_mol_mass(outtrue)
            nh.convert_mol_mass(outpred_trn)
            nh.convert_mol_mass(outtrue_trn)
            
            if flip:
                invars_trn[:,0] *=-1
                invars[:,0] *= -1
            invars_trn[:,1] *=-1
            invars[:,1] *= -1
            nh.MakeManiPlots(filename, invars_trn[:,:md.nmanivars], outtrue_trn,
                             label='true', lineage=str(order), with_cb=True, modify_ticks=True)



