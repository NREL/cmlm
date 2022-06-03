import timer
ti = timer.timer(verbose=2)
ti.start('Initialize')

import numpy as np
import pandas as pd
from sklearn.cluster       import MiniBatchKMeans
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.utils         import shuffle
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('qt4agg')

# Possible methods
# Method of partitioning Trn vs. Test :: RANDOMFILE, PATTERNFILE, RANDOM
# Method of subsampling :: NONE, RANDOM, KMEANS (only for RANDOM partioning)

##### Parameter Defaults #####
infile = ''
outfile = 'kmeans'
method  = 'random'
ratio   = 0.75
nsamples = 'all'
kmeans = 0
pattern = 'phi01.00'

##### Get paramters from the command line #####
parser=argparse.ArgumentParser(description =
                               "A tool to select training and testing data")

parser.add_argument('-i','--infile', default=infile,
                    help="Input flamelets or npz file of Pele data")

parser.add_argument('-f','--flamelet',action='store_true',
                    help="Flag if input data is from flamelets")

parser.add_argument('-s','--s3d',action='store_true',
                    help="Flag if input data is from S3D")

parser.add_argument('-v','--vfrac', default = None,
                    help="Minimum vfrac to keep for PeleC data")

parser.add_argument('-z','--npzfile',action='store_true',
                    help="Flag if input data is from an npz output from this function")

parser.add_argument('-e','--everyother',action='store_true',
                    help="Only include every other file")

parser.add_argument('-o','--outfile', default=outfile,
                    help="File to output partitioned data to")

parser.add_argument('-m','--method', default=method,
                    help="Method of partitioning data, options: random, randomfile, patternfile")

parser.add_argument('-n','--nsamples', default=nsamples,
                    help="Number of training samples or 'all' to use all")

parser.add_argument('-r','--ratio', dest='ratio', default=ratio, type=float,
                    help="Ratio of train to total samples")

parser.add_argument('-k','--kmeans', default=kmeans, type=int,
                    help="Number of kmeans clusters if doing random partitioning (0=no kmeans) and subsampling")

parser.add_argument('-p','--pattern', default=pattern, 
                    help="If using patternfile method, pattern to match to select the file(s)")

parser.add_argument('-g','--fuelspec', default='CH4', 
                    help="If using patternfile method, pattern to match to select the file(s)")

args = parser.parse_args()
if args.nsamples != 'all':
    args.nsamples = int(args.nsamples)

del infile, outfile, method, ratio, nsamples, kmeans, pattern

###### Load in Initial data ######
ti.start('Load data')
if (args.flamelet) :
    files = sorted(glob.glob(args.infile))
    if args.everyother:
        files = files[::2]
    data = {}
    tb = timer.trackerbar(len(files))
    for filename in files:
        tb.update(task=filename)
        data[filename] = pd.read_csv(filename)
    tb.finalize()

elif (args.s3d):
    import h5py
    import re
    expression = re.compile('-?\ *[0-9]\.[0-9]+[Ee][-+][0-9]+' )
    #files = sorted(glob.glob(args.infile), key = lambda x: float(re.findall(expression, x)[0]))
    files = sorted(glob.glob(args.infile))
    if args.everyother:
        files = files[::2]
    data = {}
    tb = timer.trackerbar(len(files))
    for ii,filename in enumerate(files): ### FIXME
        #print(filename, flush=True)
        tb.update(task=filename)
        with h5py.File(filename,'r') as dfile:
            if args.vfrac is not None:
                vfrac = np.array(dfile['DNS']['vfrac']).flatten()
                data[filename] = pd.DataFrame({key.replace('?','/') : np.array(dfile['DNS'][key]).flatten()[vfrac > float(args.vfrac)]
                                               for key in dfile['DNS'].keys()
                                               if key != 'planes'})

            else:
                data[filename] = pd.DataFrame({key.replace('?','/') : np.array(dfile['DNS'][key]).flatten()
                                               for key in dfile['DNS'].keys()
                                               if key != 'planes'})
            data[filename]['timestep'] = ii
    tb.finalize()
    
elif (args.npzfile):
    files = sorted(glob.glob(args.infile))
    if args.everyother:
        files = files[::2]
    data = {}
    for filename in files:
        print(filename)
        npzfile = np.load(filename,allow_pickle=True)
        data[filename] = pd.DataFrame(np.concatenate([npzfile['trndata'],npzfile['tstdata']]),
                                      columns=npzfile['columns'])
        print(npzfile['columns'])

# Other format of dta in npz file
else :
    npzfile = np.load(args.infile, allow_pickle=True)
    files = npzfile['files']
    data = npzfile['data'][()]
#raise RuntimeError
            
for ii, fi in enumerate(files):
    data[fi]['label'] = ii

###### Split data into testing and training ######

if args.method == 'randomfile' or args.method == 'patternfile':

    if args.method == 'randomfile' :
        nfiles = len(files)
        tstfiles = np.random.choice(nfiles,int((1-args.ratio)*nfiles),replace=False)
        tstfiles = np.array(files)[tstfiles]

    if args.method == 'patternfile' :
        tstfiles = np.array([fi for fi in files if args.pattern in fi] )
        
    tstdata = pd.concat([data[fi] for fi in tstfiles], ignore_index = True)
    trndata = pd.concat([data[fi] for fi in files if fi not in tstfiles], ignore_index = True)

    if args.nsamples != 'all':
        ntrn = len(trndata.index)
        if args.nsamples < ntrn:
            chosenones = np.random.choice( ntrn, args.nsamples, replace = False)
            trndata = trndata.iloc[chosenones]
        ntst = len(tstdata.index)
        nchoose = int((1-args.ratio)/args.ratio * args.nsamples) 
        if nchoose < ntst:
            chosenones = np.random.choice( ntst, nchoose, replace = False)
            tstdata = tstdata.iloc[chosenones]

elif args.method == 'random':
    data = pd.concat(data.values(), ignore_index = True, sort = True)
    ntot = len(data.index)
    if args.nsamples == 'all':
        args.nsamples = ntot
    if args.nsamples * 1/args.ratio >=  ntot:
        chosenones = np.random.choice(ntot,int(args.ratio*ntot), replace=False)
        trndata = data.iloc[chosenones]
        tstdata = data.drop(index=chosenones)
    elif args.kmeans == 0 :
        chosenones = np.random.choice(ntot,int(args.nsamples * 1/args.ratio), replace=False)
        trndata = data.iloc[chosenones[:args.nsamples]]
        tstdata = data.iloc[chosenones[args.nsamples:]]
    else :
        ti.start_subtasks()
        ti.start('Fit transform Kmeans')
        clustervars = ['O2','N2','CO','CO2','H2O','H2','OH',args.fuelspec]
        cluster_ids = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=4096).fit_predict(data[clustervars])
        chosenones_trn = []
        chosenones_tst = []
        nonzeroclusters =[]
        for cluster in range(args.kmeans):
            if len(np.where(cluster_ids == cluster)[0])>args.nsamples*0.2:
                nonzeroclusters.append(cluster)
        sampleratio = args.kmeans / len(nonzeroclusters)
        for cluster in nonzeroclusters:
            samples = shuffle(np.where(cluster_ids == cluster)[0])
            nsamples_c = len(samples)
            samplestrn = samples[:int(args.ratio*nsamples_c)]
            samplestst = samples[int(args.ratio*nsamples_c):]
            chosenones_trn.append(samplestrn[np.random.choice(len(samplestrn),int(args.nsamples*sampleratio))])
            chosenones_tst.append(samplestst[np.random.choice(len(samplestst),int((1-args.ratio)/args.ratio*args.nsamples*sampleratio))])
            print (cluster, nsamples_c, len(chosenones_trn[-1]), len(chosenones_tst[-1]))
        trndata = data.iloc[np.concatenate(chosenones_trn)]
        tstdata = data.iloc[np.concatenate(chosenones_tst)]

        #plt.scatter(trndata['N2'], trndata['CO2']+trndata['CO']+trndata['H2O']+trndata['H2'],
        #            c=cluster_ids[np.concatenate(chosenones_trn)], s=0.4)
        #plt.show()
        #plt.clf()
        #plt.scatter(trndata['N2'], trndata['CO2']+trndata['CO']+trndata['H2O']+trndata['H2'],
        #            c=trndata['label'], s=0.4)
        #plt.show()       
        ti.stop_subtasks()
else :
    raise RuntimeError('Invalid method selected')

print(trndata)
print(trndata.max( axis=1))
print(tstdata)


###### Finalize saving the data #####
ti.start('Save final data')
#np.savez_compressed(args.outfile + '.npz', trndata = trndata, tstdata = tstdata, columns=trndata.columns, files=files)
np.savez(args.outfile + '.npz', trndata = trndata, tstdata = tstdata, columns=trndata.columns, files=files)
ti.finalize()
