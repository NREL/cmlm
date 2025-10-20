"""
Create sample chemtables with dummy data for testing purposes.

Usage
-----

Invoke on the command line::

    python create_dummy_table_nd.py <input_file.toml>

Input File
----------

    The input file is a TOML format file with single section ``[table]`` that
    specifies the following:

    - ``ndim`` (int): number of dimensions in table
    - ``ngrid`` (int): number of grid points in each dimension
    - ``outfile_prefix`` (string): path to save files
"""

if __name__ == "__main__":

    import numpy as np
    import pandas as pd

    import cmlm.ctable_tools as ctt
    from cmlm.utils import TomlParmParse

    # Load inputs
    tpp = TomlParmParse("create_dummy_table_nd.toml", allow_cl_override=True)
    ndim = tpp.get("table", "ndim")
    ngrid = tpp.get("table", "ngrid")
    outfi_pref = tpp.get("table", "outfile_prefix")

    # Create empty tables
    dimnames = ["dim" + str(idim) for idim in range(ndim)]
    ctable_index = pd.MultiIndex.from_product(
        [np.linspace(0, 1, ngrid) for ii in range(ndim)], names=dimnames
    )
    data = np.zeros([ngrid**ndim, 2 + ndim])
    ctable = pd.DataFrame(
        data,
        index=ctable_index,
        columns=["RHO", "T"] + ["SRC_" + dimname for dimname in dimnames],
    )
    data_trans = np.zeros([ngrid**ndim, 2])
    ctable_trans = pd.DataFrame(
        data_trans, index=ctable_index, columns=["DIFF", "VISC"]
    )

    # Populate some dummy data
    df = ctable.reset_index()
    for idim in range(ndim):
        data = np.array(df[dimnames[idim]])
        data2 = data * data
        ctable["RHO"] += (idim + 1) * (data + 0.1 * data2)
        ctable["T"] += (idim + 1) * (data + 0.2 * data2)
        for dimname in dimnames:
            ctable["SRC_" + dimname] += (idim + 1) * (data + 0.3 * data2)
        ctable_trans["DIFF"] += (idim + 1) * (data + 0.1 * data2)
        ctable_trans["VISC"] += (idim + 1) * (data + 0.2 * data2)

    # Save tables
    ctt.write_chemtable_binary(
        f"{outfi_pref}_{ndim}dim_{ngrid}grid", ctable, "DummyData", "Pele"
    )

    ctt.write_chemtable_binary(
        f"{outfi_pref}_trans_{ndim}dim_{ngrid}grid",
        ctable_trans,
        "DummyData",
        "Pele",
    )

    # Print tables for user inspection
    print("Dummy EOS Table")
    print(ctable)
    print("\n ============ \n")
    print("Dummy Transport Table")
    print(ctable_trans)
