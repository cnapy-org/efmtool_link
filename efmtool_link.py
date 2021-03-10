import numpy
import jpype
import scipy

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

subset_compression = CompressionMethod[:]([CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting])
def compress_rat_efmtool(st, reversible, compression_method=CompressionMethod.STANDARD, remove_cr=False, tolerance=0, remove_rxns=[]):
# add keep_separate option?
# expose suppressedReactions option of StoichMatrixCompressor?
    num_met = st.shape[0]
    num_reac = st.shape[1]
    st = numpy_mat2jBigIntegerRationalMatrix(st, tolerance=tolerance)
    reversible = jpype.JBoolean[:](reversible)
    smc = StoichMatrixCompressor(compression_method)
    if remove_rxns == []:
        reacNames = jpype.JString[num_reac]
        remove_rxns = None
    else:
        reacNames = numpy.array(['R'+str(i) for i in range(num_reac)]) # set up dummy names
        remove_rxns = java.util.HashSet(reacNames[remove_rxns].tolist()) # works because of some jpype magic
        reacNames = jpype.JString[:](reacNames)
    comprec = smc.compress(st, reversible, jpype.JString[num_met], reacNames, remove_rxns)
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
    # for sparse matrices can use nonzero() here instead of iterating through everything
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
        if type(npmat) is not numpy.ndarray: # in case it is a sparse matrix
            npmat = npmat.toarray()
        jmat= DefaultBigIntegerRationalMatrix(jpype.JDouble[:](numpy.concatenate(npmat)),
                npmat.shape[0], npmat.shape[1], tolerance)
    else:
        jmat= DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(npmat), jTrue, jTrue)
    return jmat
