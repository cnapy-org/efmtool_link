import numpy
import tempfile
import jpype
import glob
import scipy
import os
import subprocess

# have to set up the path to efmtool manually here
efmtool_jar = r'E:\gwdg_owncloud\20160114_regEfmtool_3.3\regEfmtool_ori.jar'
#jpype.addClassPath(r'E:\gwdg_owncloud\efmtool-samples.jar') # merged into metabolic-efm-all.jar
#jpype.addClassPath(r'E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\lib\metabolic-efm-all.jar')
#jpype.addClassPath(r"E:\gwdg_owncloud\CNAgit\CellNetAnalyzer\code\ext\efmtool\lib")
jpype.addClassPath(efmtool_jar)
jpype.startJVM()

import jpype.imports
import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import ch.javasoft.smx.ops.Gauss as Gauss
import ch.javasoft.metabolic.compress.CompressionMethod as CompressionMethod
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import java.math.BigInteger;
jTrue = jpype.JBoolean(True)
jSystem = jpype.JClass("java.lang.System")

def null_rat_efmtool(npmat, tolerance=0):
    gauss_rat = Gauss.getRationalInstance()
    jmat = numpy_mat2jBigIntegerRationalMatrix(npmat, tolerance=tolerance)
    kn = gauss_rat.nullspace(jmat)
    return jpypeArrayOfArrays2numpy_mat(kn.getDoubleRows())

# subset_compression = CompressionMethod[:]([CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting])
def compress_rat_efmtool(st, reversible, compression_method=CompressionMethod.STANDARD, remove_cr=False, tolerance=0):
# add keep_separate option?
# expose suppressedReactions option of StoichMatrixCompressor?
    num_met = st.shape[0]
    num_reac = st.shape[1]
    st = numpy_mat2jBigIntegerRationalMatrix(st, tolerance=tolerance)
    reversible = jpype.JBoolean[:](reversible)
    smc = StoichMatrixCompressor(compression_method)
    comprec = smc.compress(st, reversible, jpype.JString[num_met], jpype.JString[num_reac], None)
    rd = jpypeArrayOfArrays2numpy_mat(comprec.cmp.getDoubleRows())
    subT = jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    if remove_cr:
        bc = basic_columns_rat(comprec.cmp.transpose())
        rd = rd[numpy.sort(bc), :] # keep row order

    return rd, subT, comprec
    
def basic_columns_rat(mx, tolerance=0): # mx is ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix
    if type(mx) is numpy.ndarray:
        mx = numpy_mat2jBigIntegerRationalMatrix(mx, tolerance=tolerance)
    row_map = jpype.JInt[mx.getRowCount()] # just a placeholder because we don't care about the row permutation here
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]

def numpy_mat2jpypeArrayOfArrays(npmat):
    rows = npmat.shape[0]
    cols = npmat.shape[1]
    jmat= jpype.JDouble[rows, cols]
    for r in range(rows):
        for c in range(cols):
            jmat[r][c]= npmat[r, c]
    return jmat

def jpypeArrayOfArrays2numpy_mat(jmat):
    rows = len(jmat)
    cols = len(jmat[0]) # assumes all rows have the same number of columns
    npmat = numpy.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            npmat[r, c]= jmat[r][c]
    return npmat

def numpy_mat2jBigIntegerRationalMatrix(npmat, tolerance=0):
    if tolerance > 0:
        jmat= DefaultBigIntegerRationalMatrix(jpype.JDouble[:](numpy.concatenate(npmat)),
                npmat.shape[0], npmat.shape[1], tolerance)
    else:
        jmat= DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(npmat), jTrue, jTrue)
    return jmat

def write_efmtool_input(st, reversible, reaction_names, metabolite_names):
    numpy.savetxt(r"stoich.txt", st)
    with open('revs.txt', 'w') as file:
        file.write(' '.join(str(x) for x in reversible))
    with open('mnames.txt', 'w') as file:
        file.write(' '.join('"' + x + '"' for x in metabolite_names))
    with open('rnames.txt', 'w') as file:
        file.write(' '.join('"' + x + '"' for x in reaction_names))

def read_efms_from_mat(folder : str) -> numpy.array:
    # taken from https://gitlab.com/csb.ethz/efmtool/
    # efmtool stores the computed EFMs in one or more .mat files. This function
    # finds them and loads them into a single numpy array.
    efm_parts : List[np.array] = []
    files_list = sorted(glob.glob(os.path.join(folder, 'efms_*.mat')))
    for f in files_list:
        mat = scipy.io.loadmat(f, verify_compressed_data_integrity=False)
        efm_parts.append(mat['mnet']['efms'][0, 0])

    return numpy.concatenate(efm_parts, axis=1)

def calculate_flux_modes(st : numpy.array, reversible, reaction_names=None, metabolite_names=None, java_executable=None):
    if java_executable is None:
        java_executable = os.path.join(str(jSystem.getProperty("java.home")), "bin", "java")
    if reaction_names is None:
        reaction_names = ['R'+str(i) for i in range(st.shape[1])]
    if metabolite_names is None:
        metabolite_names = ['M'+str(i) for i in range(st.shape[0])]
    
    curr_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as work_dir:
        print(work_dir)
        os.chdir(work_dir)
        write_efmtool_input(st, reversible, reaction_names, metabolite_names)

        cp = subprocess.Popen([java_executable,
        "-cp", efmtool_jar, "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
        '-kind', 'stoichiometry', '-arithmetic', 'double', '-zero', '1e-10',
        '-compression', 'default', '-log', 'console', '-level', 'INFO',
        '-maxthreads', '-1', '-normalize', 'min', '-adjacency-method', 'pattern-tree-minzero', 
        '-rowordering', 'MostZerosOrAbsLexMin', '-tmpdir', '.', '-stoich', 'stoich.txt', '-rev', 
        'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'matlab', 'efms.mat'],
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)
        # might there be a danger of deadlock in case an error produces a large text output that blocks the pipe?
        while cp.poll() is None:
            ln = cp.stdout.readlines(1) # blocks until one line has been read
            if len(ln) > 0: # suppress empty lines that can occur in case of external termination
                print(ln[0], end='')
        print(cp.stderr.readlines())
        os.chdir(curr_dir)
        if cp.poll() is 0:
            efms = read_efms_from_mat(work_dir)
        else:
            print("Emftool failure")
            efms = None

    return efms
