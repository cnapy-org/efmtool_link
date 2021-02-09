#%%
import efmtool_link
import numpy
    
# %%
import cobra
import cobra.util.array
# model = cobra.io.read_sbml_model(r"..\CNApy\projects\ECC2comp\ECC2comp.xml")
model = cobra.io.read_sbml_model(r"metatool_example_no_ext.xml")
#model = cobra.io.read_sbml_model(r"..\projects\iJO1366\iJO1366.xml")
rev = [r.reversibility for r in model.reactions]
stdf = cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')

#%%
kn = efmtool_link.null_rat_efmtool(stdf.values)
numpy.max(numpy.abs(stdf.values.dot(kn)))

# %% direct call to efmtool via Java 
import time
import os
import subprocess

efmtool_link.write_efmtool_input(stdf.values, numpy.array(rev, dtype=int), stdf.columns, stdf.index)
java_executable = os.path.join(str(efmtool_link.jSystem.getProperty("java.home")), "bin", "java")
#java_executable = r"C:\Program Files\AdoptOpenJDK\jdk-11.0.8.10-hotspot\bin\java"
cp = subprocess.Popen([java_executable,
"-cp", efmtool_link.efmtool_jar, "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
# "-cp", r"E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\patch.jar;E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\lib\metabolic-efm-all.jar", "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
 '-kind', 'stoichiometry', '-arithmetic', 'double', '-zero', '1e-10',
 '-compression', 'default', '-log', 'file', 'log.txt', '-level', 'INFO',
 '-maxthreads', '-1', '-normalize', 'min', '-adjacency-method', 'pattern-tree-minzero', 
 '-rowordering', 'MostZerosOrAbsLexMin', '-tmpdir', '.', '-stoich', 'stoich.txt', '-rev', 
 'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'matlab', 'efms.mat'])#,
#  stdout = subprocess.PIPE, stderr = subprocess.PIPE) #, universal_newlines=True, bufsize=1
time.sleep(0.5) # wait for efmtool to start and open the log file
with open("log.txt") as log_file:
    print(log_file.readlines())
    while cp.poll() is None:
        ln= log_file.readlines()
        if len(ln) > 0: # prevent printing lots of empty lines
            print(ln)
        else:
            time.sleep(0.1)

# %%
efms = efmtool_link.calculate_flux_modes(stdf.values, numpy.array(rev, dtype=int))
print(efms.shape)

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
        bc = efmtool_link.basic_columns_rat(x, tol)
        assert len(bc) == n
        assert numpy.linalg.matrix_rank(x[:, bc]) == n
        assert efmtool_link.null_rat_efmtool(x, tol).shape[1] == a
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
