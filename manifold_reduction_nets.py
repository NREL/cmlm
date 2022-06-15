import numpy as np
import torch
from torch import nn, optim, device, randperm
from torch.autograd import Variable
from collections import OrderedDict
import copy
import os
import sherpa
import pandas as pd
import matplotlib.pyplot as plt

# ========================================================================
# Determine the loss associated with the network structure
# Can be used to favor sparsity
class netstruct_loss:
    def __init__(self,losstype='none',lam1 = 0, ramp=0):
        self.lam1 = lam1
        self.ramp = ramp
        # Make the calc_loss function point to the right thing
        if losstype == 'none':
            self.calc_loss = self.return0
        elif losstype == 'l1l2':
            self.calc_loss = self.l1l2
        elif losstype == 'l1' :
            self.calc_loss = self.l1
        else :
            raise RuntimeError('Invalid network structure loss type')
        # create a ramp function that scales up regularization param over epochs
        if ramp == 0:
            self.calc_ramp = lambda epoch : 1.0
        else :
            self.calc_ramp = lambda epoch : min(epoch/self.ramp, 1.0)

    # No regularization
    def return0(self,model, epoch):
        return 0.0

    # L1 / L2 regularization
    def l1l2(self,model, epoch):
        loss = 0.0
        for params in model.manifold.parameters():
            for param in params:
                loss += torch.norm(param,1) / torch.norm(param,2)
        return self.lam1 * self.calc_ramp(epoch) * loss

    # L1 regularization
    def l1(self, model, epoch):
        loss = 0.0
        for params in model.manifold.parameters():
            for param in params:
                loss += torch.norm(param,1)
        return self.lam1 * self.calc_ramp(epoch) * loss
        
# ========================================================================
# Simple fully connected ANN
# input layer -> N*( FC LeakyReLU ) -> ouput
class PredictionNet(nn.Module):
    def __init__(self, Nred, H, D_out, manidef=None, Nvar=None, manibiases=None):
        super().__init__()

        self.net_type = 'PredictionNet'

        if manidef is not None:
            manidef = torch.as_tensor(manidef, dtype=torch.float)
            self.D_in, self.Nmani = manidef.shape
            self.Npass = Nred - self.Nmani

        if manibiases is None:
            manibiases = torch.zeros(self.Nmani ,dtype=torch.float)

        self.inputs = {'Nred':Nred, 'H':H, 'D_out':D_out, 'manidef':manidef, 'manibiases':manibiases, 'Nvar':Nvar}

        nh = len(H)-1

        if Nvar is None : Nvar = 0

        self.inp = nn.Linear(Nred+Nvar, H[0])

        self.nh = nh
        layers = OrderedDict()
        for i in range(self.nh):
            layers["relu" + str(i)] = nn.LeakyReLU()
            layers["bn" + str(i)] = nn.BatchNorm1d(H[i])
            layers["linear" + str(i)] = nn.Linear(H[i], H[i+1])
        self.hidden = nn.Sequential(layers)

        self.output = nn.Sequential(
            nn.LeakyReLU(), nn.BatchNorm1d(H[-1]), nn.Linear(H[-1], D_out)
        )
        
    # Calculates only the manifold variables
    def manifold(self, x):
        return torch.matmul(x[:,:self.D_in], self.inputs['manidef']) + self.inputs['manibiases']

    # Calculates manifold variables and variances if applicable
    def calc_manifold(self, x):
        D_in = self.D_in
        out1 = torch.matmul(x[:,:D_in], self.inputs['manidef']) + self.inputs['manibiases']

        # Manifold variable variances (xi_var)
        if self.inputs['Nvar'] is not None:
            out2 = x[:,D_in:D_in + D_in**2].reshape(-1,D_in,D_in)
            out2 = self.manifold(out2).transpose(1,2)
            out2 = self.manifold(out2)
            out2 = torch.diagonal(out2,dim1=1,dim2=2) - out1**2
            out2 = out2[:,:self.inputs['Nvar']]

            # Passed through variables (passvars)
            out3 = x[:,D_in+D_in**2:]
            return torch.cat((out1, out2, out3),1)
        
        else:
            out3 = x[:,D_in:]
            return torch.cat((out1, out3),1)
        
    # Wrapper around forward for compatibility reasons:
    def mapfrom_manifold(self,x):
        out = self.forward(x)
    
    def forward(self, x):
        out = self.inp(x)
        out = self.hidden(out)
        out = self.output(out)
        return out

    def get_inputs(self):
        return copy.deepcopy(self.inputs)

# ========================================================================
# Create a prediction net that takes in and spits out unscaled data
#
def unscale_prediction_net(PredNet, scalers):
    # Get parameters from existing net that uses scaled inputs/outputs
    params = PredNet.get_inputs()
    statedict = PredNet.state_dict()

    # Change the input parameters so raw inputs can be provided
    print(params['Nvar'])
    if params['Nvar'] is not None and params['Nvar'] != 0:
        raise RuntimeError("Unscaling prediction nets is not yet supported when variances are used")
    if params['Nred'] != params['manidef'].shape[1]:
        raise RuntimeError("Unscaling prediction nets is not yet supported when pass through variables are used")
    params['manidef']    = (params['manidef'].T / torch.Tensor(scalers['inp'].scale_)).T
    params['manibiases'] = -torch.matmul(params['manidef'].T,torch.Tensor(scalers['inp'].mean_))

    # Change output layer parameters so unscaled outputs are delivered
    statedict['output.2.weight'] = (statedict['output.2.weight'].T * torch.Tensor(scalers['out'].scale_)).T
    statedict['output.2.bias'] *= torch.Tensor(scalers['out'].scale_)
    statedict['output.2.bias'] += torch.Tensor(scalers['out'].mean_)

    # Create and return a network with the modified parameters
    UnscaledNet = PredictionNet(**params)
    UnscaledNet.load_state_dict(statedict)
    UnscaledNet.eval()
    return UnscaledNet
    
# ========================================================================
# Co-optimied Machine Learned Manifolds network structure
# input layer -> linear manifold reduction layer ->  N*( FC LeakyReLU ) -> ouput
# Npass variables at the end of the input vector are passed firectly to the hidden layers
# The variances of the first Nvar input variables are also calculated (by default no Variances)
# if Nvar= None, inputs vector to the model should not have any variances.
# if Nvar = an integer, input vector must have all variances
class ManifoldReductionNet(nn.Module):
    def __init__(self, D_in, Nred, H, D_out, Npass=0, Nvar =None):
        super().__init__()
        
        self.net_type = 'ManifoldReductionNet'
        
        # Set things up for the right number of variances
        self.calc_manifold = self.calc_manifold_wvar
        NvarTmp = Nvar
        if Nvar is None:
            self.calc_manifold = self.calc_manifold_novar
            NvarTmp = 0
        elif Nvar is 'all':
            Nvar = Nred - Npass
        elif Nvar > Nred - Npass or Nvar < 0 :
            raise RuntimeError("Invalid number of variances specified")
            
        self.inputs = {'D_in':D_in, 'Nred':Nred, 'H':H, 'D_out':D_out, 'Npass':Npass, 'Nvar':Nvar}
        self.nh = len(H)-1

        # Set up the layers
        self.manifold = nn.Linear(D_in, Nred-Npass, bias=False)
        
        self.inp = nn.Linear(Nred + NvarTmp,H[0])
        
        layers = OrderedDict()
        for i in range(self.nh):
            layers["relu" + str(i)] = nn.LeakyReLU()
            layers["bn" + str(i)] = nn.BatchNorm1d(H[i])
            layers["linear" + str(i)] = nn.Linear(H[i], H[i+1])
        self.hidden = nn.Sequential(layers)

        self.output = nn.Sequential(
            nn.LeakyReLU(), nn.BatchNorm1d(H[-1]), nn.Linear(H[-1], D_out)
        )

    def calc_manifold_novar(self,xin):
        D_in = self.inputs['D_in']
        out1= self.manifold(xin[:,:D_in]) # Manifold variables
        # No variances
        out3 = xin[:,D_in:] # Pass through variables
        return torch.cat((out1, out3),1)
            
    def calc_manifold_wvar(self,xin):
        # xin: Yfilt, YcorrelationMatrix, passvars
        # output is manifold: xi_filt, passvars, xi_var
        nmani = self.inputs['Nred'] - self.inputs['Npass']
        npass = self.inputs['Npass']
        D_in = self.inputs['D_in']
        nsamples = xin.shape[0]

        # Manifold variables (xi_filt)
        out1= self.manifold(xin[:,:D_in])

        # Manifold variable variances (xi_var)
        out2 = xin[:,D_in:D_in + D_in**2].reshape(-1,D_in,D_in)
        out2 = self.manifold(out2).transpose(1,2)
        out2 = self.manifold(out2)
        out2 = torch.diagonal(out2,dim1=1,dim2=2) - out1**2
        out2 = out2[:,:self.inputs['Nvar']]

        # Passed through variables (passvars)
        out3 = xin[:,D_in+D_in**2:]
        
        return torch.cat((out1, out2, out3),1)

    def mapfrom_manifold(self,xmani):
        out = self.inp(xmani)
        out = self.hidden(out)
        out = self.output(out)
        return out
        
    def forward(self, xin):
        out = self.calc_manifold(xin)
        out = self.mapfrom_manifold(out)
        return out

    def get_inputs(self):
        return copy.deepcopy(self.inputs)
    
# ========================================================================
# Extract the prediction net (nonlinear portion) from a manifold reduction net
def manifold2prediction(ManiNet):
    
    statedict = ManiNet.state_dict()
    ManiDef = copy.deepcopy(statedict['manifold.weight'])
    for key in list(statedict.keys()):
        if 'manifold' in key: del statedict[key]

    params = ManiNet.get_inputs()
    del(params['D_in'])
    del(params['Npass'])
    params['manidef'] = ManiDef.T
    PredNet = PredictionNet(**params)
    PredNet.load_state_dict(statedict)
    PredNet.eval()
    
    return PredNet

# ========================================================================
# Use a prediction net to create a reduction net
# 
def get_manifold_net(PredNet, reinit=False, Npass = 0):

    params = PredNet.get_inputs()
    ManiDef = params.pop('manidef')
    params.pop('manibiases')
    nvarin, nmanivar = ManiDef.shape
    if nmanivar + Npass != params['Nred']:
        raise RuntimeError("Dimension of manifold does not match input dimension of Prediction net")
    params['D_in'] = nvarin
    params['Npass'] = Npass
    ManiNet = ManifoldReductionNet(**params)
    statedict = ManiNet.state_dict()
    statedict['manifold.weight'] = torch.as_tensor(ManiDef.T,dtype=torch.float)
    if not reinit:
        statedict_Pred = PredNet.state_dict() 
        for key in statedict_Pred.keys():
            statedict[key] = statedict_Pred[key]
    ManiNet.load_state_dict(statedict)
    
    return ManiNet

# ========================================================================
# Train a network
def train_net(XTrain, YTrain, Xval, Yval, model,
              dev = device("cpu"),
              lossfunc = nn.MSELoss(),
              wgtlossfunc = netstruct_loss('none').calc_loss,
              batchsize = 1024,
              nepochs = 1000,
              learning_rate= 0.001,
              reduce_lr = 0,
              stop_no_progress = 0,
              optimizer = None,
              ondev=False,
              save_chk=False,
              increase_bs=False,
              return_full_hist=False):

    # Create Tensors from our numpy arrays + put everything on right device
    if not ondev:
        Xt = Variable(torch.as_tensor(XTrain,dtype=torch.float).to(device=dev))
        Yt = Variable(torch.as_tensor(YTrain,dtype=torch.float).to(device=dev))
        Xv = Variable(torch.as_tensor(Xval,dtype=torch.float).to(device=dev))
        Yv = Variable(torch.as_tensor(Yval,dtype=torch.float).to(device=dev))
        model.to(device=dev)
        lossfunc = lossfunc.to(device = dev)
    else:
        Xt = XTrain
        Yt = YTrain
        Xv = Xval
        Yv = Yval

    # Train the model
    nsamples = Xt.size()[0]
    nbatches = nsamples // batchsize
    minloss = np.inf
    n_no_prog = 0
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    if return_full_hist:
        hist = dict(zip(['val_loss','trn_loss','wgt_loss','lr','bs'],[[],[],[],[],[]]))
            
    for epoch in range(nepochs):
        
        model.train()
        permutation = randperm(nsamples)

        # Optimize individually for each random batch
        for batchstart in range(0,nsamples,batchsize):
            #print(batchstart, batchsize)
            indices = permutation[batchstart : batchstart + batchsize]
            batch_x, batch_y = Xt[indices], Yt[indices]
            ypred = model(batch_x)
            loss = lossfunc(ypred, batch_y) + wgtlossfunc(model,epoch)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute the validation loss and print
        model.eval()
        val_loss = lossfunc(model(Xv), Yv).item() 
        trn_loss = lossfunc(model(Xt), Yt).item()
        wgt_loss = wgtlossfunc(model,epoch) 

        if return_full_hist:
            hist['val_loss'].append(float(val_loss))
            hist['trn_loss'].append(float(trn_loss))
            hist['wgt_loss'].append(float(wgt_loss))
            hist['lr'].append(float(learning_rate))
            hist['bs'].append(int(batchsize))
            
        print("Epoch [{0:d}/{1:d}], Training loss {2:.4e}, Validation loss: {3:.4e}, Weight loss: {4:.4e}".format(
                epoch + 1, nepochs, (trn_loss), (val_loss), wgt_loss),flush=True)
        
        # Stop or reduce learning rate if reduction of loss has stalled
        if stop_no_progress > 0 :
            # If improving, do nothing
            if val_loss < minloss:
                minloss = val_loss
                mintrnloss = trn_loss
                n_no_prog = 0
                best_optim = copy.deepcopy(optimizer.state_dict())
                best_model = copy.deepcopy(model.state_dict())
            # If no improvement in a long time, revert to best model, then stop or reduce learning rate
            elif n_no_prog >= stop_no_progress:
                if reduce_lr > 0:
                    learning_rate = learning_rate/10.0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    print ('  ... Not making progress, reducing learning rate to {}'.format(learning_rate))
                    if increase_bs:
                        batchsize *= 2
                        print ('  ... and increasing batchsize to {}'.format(batchsize))
                    n_no_prog = 0
                    reduce_lr = reduce_lr - 1
                else:
                    print ('  ... Not making progress, stopping training')
                    break
            # If not improving, count how long since last improvement
            else:
                n_no_prog += 1
        else :
            if val_loss < minloss:
                minloss = val_loss
                mintrnloss = trn_loss
                best_optim = copy.deepcopy(optimizer.state_dict())
                best_model = copy.deepcopy(model.state_dict())

        if save_chk is not False:
            save_checkpoint(model, optimizer, os.path.join(save_chk, 'chk' +str(epoch) + '.npz') , verbose= False)
                
    optimizer.load_state_dict(best_optim)
    model.load_state_dict(best_model)
        
    if return_full_hist:
        return mintrnloss, minloss, hist
    else:
        return mintrnloss, minloss

def load_checkpoint(model = None, optimizer = None, filename = None, verbose = True, generate_new = False):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        if verbose: print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

        # Optionally, generate a totally new model rather than modifying the existing one.
        if generate_new:
            if verbose: print("  => generating new model from scratch")
            if checkpoint['net_type'] == 'PredictionNet':
                model = PredictionNet(**checkpoint['model_inputs'])
            elif checkpoint['net_type'] == 'ManifoldReductionNet':
                model = ManifoldReductionNet(**checkpoint['model_inputs'])
            else:
                raise RuntimeError('Invalid model type specified')
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
        if verbose: print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer

def save_checkpoint(model, optimizer, filename, verbose = True):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'model_inputs':model.inputs,
             'net_type':model.net_type}
    if verbose: print("=> saving checkpoint '{}'".format(filename))
    torch.save(state, filename)
    return 0

# Creates a network with the same structure as the input model
# but with a new set of parameters
# but for ManifoldReductionNets, the manifold definition layer is kept.
def copy_net(model, keep_manidef=True):

    if model.net_type == 'PredictionNet':
        new_model = PredictionNet(**model.inputs)
        
    elif model.net_type == 'ManifoldReductionNet':
        new_model = ManifoldReductionNet(**model.inputs)
        if keep_manidef:
            new_statedict = new_model.state_dict()
            old_statedict = model.state_dict()
            new_statedict['manifold.weight'] = old_statedict['manifold.weight']
            new_model.load_state_dict(new_statedict)
            
    elif model.net_type == 'PCANet':
        new_model = PCANet(**model.inputs)

    elif model.net_type == 'FilteredManifoldReductionNet':
        new_model = FilteredManifoldReductionNet(**model.inputs)
        
    else:
        raise RuntimeError('Cannot copy this type of network')
        
    return new_model

# ========================================================================
# Train a network
def train_net_sherpa(XTrain, YTrain, Xval, Yval, model_in,
                     dev = device("cpu"),
                     lossfunc = nn.MSELoss(),
                     wgtlossfunc = netstruct_loss('none').calc_loss,
                     batchsize = [512, 1024, 2048, 4096, 8192, 16384, 32768],
                     learning_rate= [1e-6, 1e-2],
                     nepochs = 1,
                     ngenerations = 100,
                     nsiblings = 10,
                     stop_no_progress = 0, reduce_lr =0,
                     savepath = './saved_networks/',
                     plot_dashboard ='dashboard',
                     save_chk =False,
                     increase_bs=False):

    # make the output directory:
    if not os.path.exists(savepath): os.makedirs(savepath)
    
    parameters = [sherpa.Continuous('lr', learning_rate, scale='log'),
                  sherpa.Ordinal('batchsize', batchsize)]
    algorithm = sherpa.algorithms.PopulationBasedTraining(population_size = nsiblings,
                                                          num_generations = ngenerations,
                                                          perturbation_factors = (0.1,1,10) )
    if plot_dashboard is not None:
        fig = plt.figure(plot_dashboard,figsize=[12,4])
        axTL = plt.subplot(221)
        axTR = plt.subplot(222, sharex = axTL, sharey=axTL)
        axBL = plt.subplot(223, sharex = axTL)
        axBR = plt.subplot(224, sharex = axTL)
        axBL.set_xlabel('Generation')
        axBR.set_xlabel('Generation')
        axTL.set_ylabel('Training Loss')
        axTR.set_ylabel('Validation Loss')
        axBL.set_ylabel('Learning Rate')
        axBR.set_ylabel('Batch Size')
        axTL.set_yscale('log')
        axTR.set_yscale('log')
        axBL.set_yscale('log')
        axBR.set_yscale('log')
        cmap = plt.get_cmap('tab20')
        
    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)

    TL={}; VL={}; BS={}; LR={}; GEN ={}; hist ={};

    # put all the data on the device
    Xt = Variable(torch.as_tensor(XTrain,dtype=torch.float).to(device=dev))
    Yt = Variable(torch.as_tensor(YTrain,dtype=torch.float).to(device=dev))
    Xv = Variable(torch.as_tensor(Xval,dtype=torch.float).to(device=dev))
    Yv = Variable(torch.as_tensor(Yval,dtype=torch.float).to(device=dev))
    lossfunc = lossfunc.to(device = dev)
        
    for trial in study:
        # Get the parameters
        generation   = trial.parameters['generation']
        load_from    = trial.parameters['load_from']
        save_to      = trial.parameters['save_to']
        learningrate = trial.parameters['lr']
        batchsize    = int(trial.parameters['batchsize'])
        print ("Starting trial {} in generation {}:     lr = {:8.4e}   bs = {:6n}"
               .format(trial.id, generation,learningrate,batchsize))
        
        # Get the model and optimizer and load if necessary
        model = copy_net(model_in)
        optimizer = optim.Adam(model.parameters(), lr=learningrate, amsgrad=True)
        if load_from != "":
            load_checkpoint(model, optimizer,
                            os.path.join(savepath, load_from + '.npz'), verbose=False )

        GEN[trial.id] = generation; BS[trial.id] = batchsize; LR[trial.id] = learningrate
        model.to(device=dev)
        if save_chk is not False: save_chk = savepath
        TL[trial.id], VL[trial.id], hist[trial.id] = train_net(Xt, Yt, Xv, Yv, model, 
                                                     batchsize=batchsize, nepochs=nepochs, dev=dev,
                                                     lossfunc = lossfunc, learning_rate=learningrate,
                                                     wgtlossfunc = wgtlossfunc, ondev=True,
                                                     stop_no_progress=stop_no_progress, reduce_lr = reduce_lr,
                                                     save_chk = save_chk, increase_bs=increase_bs,
                                                     return_full_hist = True)
        save_checkpoint(model, optimizer, os.path.join(savepath, save_to + '.npz') , verbose= False)
        hist[trial.id] = pd.DataFrame(hist[trial.id])
        hist[trial.id].to_csv(os.path.join(savepath, save_to + '.csv'))
        study.add_observation(trial=trial, objective=VL[trial.id], iteration=int(generation)) 
        study.finalize(trial=trial)
        study.save(output_dir = savepath)
 
        if plot_dashboard is not None:
            child = trial.id
            
            hist[child].index += (GEN[child]-1)*nepochs+1 
            if load_from != '':
                ancestor = float(trial.parameters['lineage'].split(',')[0])
                parent = int(load_from)
                hist[child].loc[GEN[parent]*nepochs] = hist[parent].loc[GEN[parent]*nepochs]
                hist[child] = hist[child].sort_index()
            else:
                ancestor = child

            colorid=float(ancestor) / (nsiblings + 1) + 0.5/(nsiblings+1)
            axTL.plot(hist[child].index,hist[child]['trn_loss'], color=cmap(colorid))
            axTR.plot(hist[child].index,hist[child]['val_loss'], color=cmap(colorid))
            axBL.plot(hist[child].index,hist[child]['lr'], color=cmap(colorid))
            axBR.plot(hist[child].index,hist[child]['bs'], color=cmap(colorid))
            plt.savefig(savepath + '/' + plot_dashboard + '.png')          
        if os.path.isfile('stop') :
            os.remove('stop')
            break 

    best_trial = study.get_best_result()
    best_id = best_trial['Trial-ID']
    with open(os.path.join(savepath, 'best_trial.txt'),'w') as f:
        f.write("{}".format(best_id))
    
    load_checkpoint(model_in, optimizer,
                    os.path.join(savepath, str(best_trial['Trial-ID']) + '.npz'), verbose=False )  
    
    return TL[best_id], VL[best_id]
    
    
    
    
    
