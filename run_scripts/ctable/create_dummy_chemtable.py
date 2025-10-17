import cmlm.ctable_tools as ctt
import cmlm.utils.TomlParmParse
import pandas as pd
import numpy as np

# Load inputs
tpp = TomlParmParse("create_dummy_chemtable.toml", allow_cl_override=True)
ndim = tpp.get("table","ndim")
ngrid = tpp.get("table","ngrid")
outfi_pref = tpp.get("table","outfile_prefix")

# Create empty tables
dimnames = ["dim"+str(idim) for idim in range(ndim)]
ctable_index = pd.MultiIndex.from_product([np.linspace(0,1,ngrid) for ii in range(ndim)], names = dimnames)
data = np.zeros([ngrid ** ndim, 2 + ndim])
ctable = pd.DataFrame(data, index=ctable_index,
                      columns=['RHO','T']+["SRC_"+dimname for dimname in dimnames])
dataT = np.zeros([ngrid ** ndim, 2])
ctableT = pd.DataFrame(dataT, index=ctable_index,
                       columns=['DIFF','VISC'])

# Populate some dummy data
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

# Save tables
ctt.write_chemtable_binary(outfi_pref+str(ndim)+'dim_'+str(ngrid)+'grid',
                           ctable, 'DOGS', 'Pele')

ctt.write_chemtable_binary(outfi_pref+'_trans_'+str(ndim)+'dim_'+str(ngrid)+'grid',
                           ctableT, 'DOGS', 'Pele')

# Print tables for user inspection
print("Dummy EOS Table")
print(ctable)
print("\n ============ \n\n")
print("Dummy Transport Table")
print(ctableT)
