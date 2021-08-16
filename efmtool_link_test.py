#%%
import efmtool_link.efmtool_intern as efmtool_intern
import numpy
    
# %%
import cobra
import cobra.util.array
# model = cobra.io.read_sbml_model(r"..\CNApy\projects\ECC2comp\ECC2comp.xml")
model = cobra.io.read_sbml_model(r"metatool_example_no_ext.xml")
#model = cobra.io.read_sbml_model(r"..\projects\iJO1366\iJO1366.xml")
rev = numpy.array([int(r.reversibility) for r in model.reactions])
stdf = cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')

#%%
kn = efmtool_intern.null_rat_efmtool(stdf.values)
numpy.max(numpy.abs(stdf.values.dot(kn)))

# %% direct call to efmtool via Java 
import time
import os
import subprocess
import efmtool_link.efmtool_extern as efmtool_extern

efmtool_extern.write_efmtool_input(stdf.values, numpy.array(rev, dtype=int), stdf.columns, stdf.index)
java_executable = os.path.join(str(efmtool_intern.jSystem.getProperty("java.home")), "bin", "java")
#java_executable = r"C:\Program Files\AdoptOpenJDK\jdk-11.0.8.10-hotspot\bin\java"
if os.path.isfile('log.txt'):
    os.remove('log.txt')
cp = subprocess.Popen([java_executable,
"-cp", efmtool_intern.efmtool_jar, "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
# "-cp", r"E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\patch.jar;E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\lib\metabolic-efm-all.jar", "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
 '-kind', 'stoichiometry', '-arithmetic', 'double', '-zero', '1e-10',
 '-compression', 'default', '-log', 'file', 'log.txt', '-level', 'INFO',
 '-maxthreads', '-1', '-normalize', 'min', '-adjacency-method', 'pattern-tree-minzero', 
 '-rowordering', 'MostZerosOrAbsLexMin', '-tmpdir', '.', '-stoich', 'stoich.txt', '-rev', 
 'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'binary-doubles', 'efms.bin'])#, # Java uese big endian
#  'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'matlab', 'efms.mat'])#,
#  stdout = subprocess.PIPE, stderr = subprocess.PIPE) #, universal_newlines=True, bufsize=1
# time.sleep(0.5) # wait for efmtool to start and open the log file
while cp.poll() is None: # wait for log file to appear
    print("Wating for log.txt")
    time.sleep(0.1)
    if os.path.isfile('log.txt'):
        break
with open("log.txt") as log_file:
    # print(log_file.readlines())
    while cp.poll() is None:
        ln= log_file.readlines()
        if len(ln) > 0: # prevent printing lots of empty lines
            print("".join(ln))
        else:
            time.sleep(0.1)
# %% read efm binary format
with open('efms.bin', 'rb') as fh:
    num_efm = numpy.fromfile(fh, dtype='>i8', count=1)[0]
    num_reac = numpy.fromfile(fh, dtype='>i4', count=1)[0]
    numpy.fromfile(fh, numpy.byte, count=1) # skip binary flag (boolean written as byte)
    efm = numpy.fromfile(fh, dtype='>d', count=num_reac*num_efm) # '<d' or 'float64' will not work
numpy.max(numpy.abs(stdf.values@efm.reshape((num_reac, num_efm), order='F')))
numpy.max(numpy.abs(stdf.values@efm.reshape((num_efm, num_reac), order='C').transpose()))

# %% open as memory map
efm_mmap = numpy.memmap('efms.bin', mode='r+', dtype='>d', offset=13, shape=(num_reac, num_efm), order='F')
numpy.max(numpy.abs(stdf.values@efm_mmap))

# %%
import efmtool_link.efmtool_extern as efmtool_extern
import numpy
import pickle
efms = efmtool_extern.calculate_flux_modes(stdf.values, rev) #numpy.array(rev, dtype=int))
# efmtool results are all irreversible (both directions of a reversible mode are present)
# numpy.any(efms.fv_mat[:, rev == 0], axis=1)
numpy.max(numpy.abs(stdf.values@efms))
# %%
fvc = FluxVectorContainer(efms.transpose(), model.reactions.list_attr('id'))
fvc.save('test')
# fvm = FluxVectorMemmap('test.zip')

# %%
import tempfile
tempfile.tempdir = r"E:\cnapy_tmp"

# %%
import efmtool_link.efmtool_extern as efmtool_extern
rev2 = rev
rev2[0:5] = 1
wd = efmtool_extern.calculate_flux_modes(stdf.values, numpy.array(rev2, dtype=int), return_work_dir_only=True)
# with open(os.path.join(wd.name, 'efms.bin'), 'rb') as fh:
#     num_efm = numpy.fromfile(fh, dtype='>i8', count=1)[0]
#     num_reac = numpy.fromfile(fh, dtype='>i4', count=1)[0]
from cnapy.flux_vector_container import FluxVectorMemmap, FluxVectorContainer
# efms = FluxVectorMemmap('efms.bin', (num_reac, num_efm), model.reactions.list_attr('id'), offset=13, containing_temp_dir=wd)
efms = FluxVectorMemmap('efms.bin', model.reactions.list_attr('id'), containing_temp_dir=wd)
del wd
# numpy.any(efms.fv_mat[:, rev == 0], axis=1)
# efms.fv_mat[numpy.any(efms.fv_mat[:, rev == 0], axis=1) == False]
is_irrev_efm = numpy.any(efms.fv_mat[:, rev == 0], axis=1)
# is_rev_emf = numpy.any(efms.fv_mat[:, rev == 0], axis=1) == False
# rev_emfs_idx = numpy.nonzero(is_rev_emf)[0]
rev_emfs_idx = numpy.nonzero(is_irrev_efm == False)[0]
if len(rev_emfs_idx) > 0:
    # numpy.unique(efms.fv_mat[rev_emfs_idx, :] != 0., axis = 0, return_index=True)
    del_idx = rev_emfs_idx[numpy.unique(efms.fv_mat[rev_emfs_idx, :] != 0., axis = 0, return_index=True)[1]]
    # is_rev_emf = numpy.delete(is_rev_emf, del_idx)
    is_irrev_efm = numpy.delete(is_irrev_efm, del_idx)
    efms = FluxVectorContainer(numpy.delete(efms.fv_mat, del_idx, axis=0), reac_id=efms.reac_id, irreversible=is_irrev_efm)

# %%
efms.save('test.npz')
fvc = FluxVectorContainer('test.npz')
efms[0] == fvc[0]

# %% 
import pickle
# with open(os.path.join(wd.name, 'efms.bin'), 'ab') as fh:
#     pickle.dump()
#     # numpy.array(model.reactions.list_attr('id')).tofile(fh) # only constant length strings appear possible
import zipfile
with zipfile.ZipFile('efms.zip', mode='w') as zf: # mode w destroys an existing file, in-place zipping appears not to be possible
    # does not overwrite an existing file, for this use mode 'w'
    zf.write(os.path.join(wd.name, 'efms.bin'), arcname='efms.bin')
    zf.writestr('efm_info', pickle.dumps({'num_reac': num_reac, 'num_efm': num_efm, 'reac_id': model.reactions.list_attr('id')}))

# %% reaction flipping
import cobra
from cobra import Model, Reaction, Metabolite
import efmtool_link.efmtool4cobra as efmtool4cobra
model = Model('subsets')
model.add_metabolites([Metabolite('A', compartment='c')])
r1 = Reaction('R1')
r2 = Reaction('R2')
model.add_reactions([r1, r2])
r1.lower_bound = 0.
r1.upper_bound = 1000.
r1.add_metabolites({'A': 1.0})
print(r1.reversibility) # irreversible forward, produces A
# r2.lower_bound = 0.
# r2.upper_bound = 1000.
# r2.add_metabolites({'A': -1.0})
r2.lower_bound = -1000.
r2.upper_bound = 0.
r2.add_metabolites({'A': 1.0})
print(r2.reversibility) # irreversible backward, consumes A
cobra.io.write_sbml_model(model, r"irrev_back_test.xml")
# model.optimize()
# cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')
subT = efmtool4cobra.compress_model_sympy(model)

# %% test
def test2(n, a, m=0, dec=None, tol=0):
    # n = 20 # number of columns
    # m = 1 # n+m: number of rows
    # a = n # added linearly dependen columns
    # tol = 1e-10
    # dec= 3
    for i in range(100):
        x = numpy.random.rand(n+m, n)
        if dec is not None:
            numpy.round(x, out=x, decimals=dec)
        #x = numpy.random.randint(0, 10, ((n+m, n)))
        #print(numpy.linalg.matrix_rank(x))
        assert numpy.linalg.matrix_rank(x) == n
        y = numpy.random.rand(n, a)
        if dec is not None:
            numpy.round(y, out=y, decimals=dec)
        #print(x @ y)
        x = numpy.hstack((x, x @ y)) # because of x*y a lower tolerance than 1e-dec is needed
        #x = numpy.hstack((x, x.dot(numpy.random.randint(0, 10, (n, n)))))
        p = numpy.random.permutation(n+a)
        x = x[:, p]
        bc = efmtool_intern.basic_columns_rat(x, tol)
        assert len(bc) == n
        assert numpy.linalg.matrix_rank(x[:, bc]) == n
        assert efmtool_intern.null_rat_efmtool(x, tol).shape[1] == a
        #print(numpy.linalg.matrix_rank(x))
        #print(null_rat_efmtool(x).shape)
        #print(numpy.where(p >= n))
        #print(bc)
        assert numpy.linalg.matrix_rank(x) == n

# %% some tests take very long
test2(20, 20) # OK
test2(20, 20, m=1) # fails
test2(20, 20, m=1, dec=3, tol= 1e-10) # OK
test2(20, 20, m=1, dec=2, tol= 1e-7) # OK
test2(20, 20, m=1, dec=4, tol= 1e-12) # fails
test2(20, 20, m=1, dec=4, tol= 1e-13) # OK
test2(20, 20, m=1, dec=4, tol= 1e-14) # OK
test2(20, 20, m=1, dec=4, tol= 1e-15) # fails
test2(20, 20, m=1, dec=5, tol= 1e-15) # fails, also for lower tolerances
test2(9, 9, m=1, dec=5, tol= 1e-15) # OK
