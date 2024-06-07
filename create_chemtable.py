import ctable_tools as ctt
import pandas as pd
import numpy as np
import sys

ndim = int(sys.argv[1])
ngrid = int(sys.argv[2])

dimnames = ["dim"+str(idim) for idim in range(ndim)]
ctable_index = pd.MultiIndex.from_product([np.linspace(0,1,ngrid) for ii in range(ndim)], names = dimnames)
data = np.zeros([ngrid ** ndim, 2 + ndim])
ctable = pd.DataFrame(data, index=ctable_index,
                      columns=['RHO','T']+["SRC_"+dimname for dimname in dimnames])
dataT = np.zeros([ngrid ** ndim, 2])
ctableT = pd.DataFrame(dataT, index=ctable_index,
                       columns=['DIFF','VISC'])

df = ctable.reset_index()
for idim in range(ndim):
    data = np.array(df[dimnames[idim]])
    data2 = data*data
    ctable['RHO'] += (idim+1)*(data + 0.1*data2)
    ctable['T'] += (idim+1)*(data + 0.2*data2)
    for dimname in dimnames:
        ctable['SRC_'+dimname] += (idim+1)*(data + 0.3*data2)
    ctableT['DIFF'] += (idim+1)*(data + 0.1*data2)
    ctableT['VISC'] += (idim+1)*(data + 0.2*data2)

ctt.write_chemtable_binary('peletable_'+str(ndim)+'dim_'+str(ngrid)+'grid',
                           ctable, 'DOGS', 'Pele')

ctt.write_chemtable_binary('peletable_trans_'+str(ndim)+'dim_'+str(ngrid)+'grid',
                           ctableT, 'DOGS', 'Pele')

print(ctable)

print(ctableT)
