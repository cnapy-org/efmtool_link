import efmtool_link
import numpy
import cobra
import sympy
import scipy.sparse
import jpype
import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import java.math.BigInteger as BigInteger
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import ch.javasoft.math.BigFraction as BigFraction

# %% make a compressed model 
# subT can be used for conversion between its result and the full model
def compress_model(model):
    # the subsets are in the columns of subset_matrix, its rows correspond to the reactions
    subset_matrix = efmtool_link.compress_rat_efmtool(cobra.util.array.create_stoichiometric_matrix(model, array_type='dok'),
                     [r.reversibility for r in model.reactions], remove_cr=False, compression_method=efmtool_link.subset_compression)[1]
    compressed_model = model.copy()
    del_rxns = numpy.logical_not(numpy.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            compressed_model.reactions[rxn_idx[i]] *= subset_matrix[rxn_idx[i], j]
        compressed_model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        compressed_model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j]
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            compressed_model.reactions[rxn_idx[0]] += compressed_model.reactions[rxn_idx[i]]
            # the stoichiometries calculated here are less exact than those in rd from compress_rat_efmtool
            # therefore when calulating e.g. conservation relations of the compressed model a non-zero tolerance
            # may be needed to recover the conservation relations
            if compressed_model.reactions[rxn_idx[i]].lower_bound == 0:
                compressed_model.reactions[rxn_idx[0]].lower_bound = 0
            del_rxns[rxn_idx[i]] = True
    del_rxns = numpy.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        compressed_model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    # does not remove the conservation relations
    subT = numpy.zeros((len(model.reactions), len(compressed_model.reactions)))
    for j in range(subT.shape[1]):
        subT[compressed_model.reactions[j].subset_rxns, j] = compressed_model.reactions[j].subset_stoich
    # adapt compressed_model name
    return compressed_model, subT

def compress_model_sympy(model):
    # the subsets are in the columns of subset_matrix, its rows correspond to the reactions
    compr_model = model.copy()
    num_met = len(compr_model.metabolites)
    num_reac = len(compr_model.reactions)
    stoich_mat = DefaultBigIntegerRationalMatrix(num_met, num_reac)
    # reversible = jpype.JBoolean[num_reac]
    reversible = jpype.JBoolean[:]([r.reversibility for r in compr_model.reactions])
    for i in range(num_reac):
        # have to use _metabolites because metabolites gives only a copy
        for k, v in compr_model.reactions[i]._metabolites.items():
            if type(v) is float or type(v) is int:
                if type(v) is int or v.is_integer():
                    # v = int(v)
                    v = sympy.Rational(v) # for simplicity
                    # compr_model.reactions[i]._metabolites[k] = int(v)
                else:
                    v = sympy.Rational(v)
                    # compr_model.reactions[i]._metabolites[k] = sympy.Rational(v) #c # only changes coefficient in the model, not in the solver
                compr_model.reactions[i]._metabolites[k] = v
            elif type(v) is not sympy.Rational:
                TypeError
            n, d = sympyRat2jBigIntegerPair(v)
            # does not work altought there is a public void setValueAt(int row, int col, BigInteger numerator, BigInteger denominator) method
            # leads to kernel crash directly or later
            # stoic_mat.setValueAt(compr_model.metabolites.index(k.id), i, n, d)
            stoich_mat.setValueAt(compr_model.metabolites.index(k.id), i, BigFraction(n, d))
            # reversible[i] = compr_model.reactions[i].reversibility # somehow makes problems with the smc.compress call
    
    smc = StoichMatrixCompressor(efmtool_link.subset_compression)
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], jpype.JString[num_reac], None)

    subset_matrix= efmtool_link.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = numpy.logical_not(numpy.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            compr_model.reactions[rxn_idx[i]] *= jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(rxn_idx[i], j)) #subset_matrix[rxn_idx[i], j]
        compr_model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        compr_model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j] # use rationals here?
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            compr_model.reactions[rxn_idx[0]] += compr_model.reactions[rxn_idx[i]]
            # the stoichiometries calculated here are less exact than those in rd from compress_rat_efmtool
            # therefore when calulating e.g. conservation relations of the compressed model a non-zero tolerance
            # may be needed to recover the conservation relations
            if compr_model.reactions[rxn_idx[i]].lower_bound == 0:
                compr_model.reactions[rxn_idx[0]].lower_bound = 0
            del_rxns[rxn_idx[i]] = True
    del_rxns = numpy.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        compr_model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    # does not remove the conservation relations
    subT = numpy.zeros((len(model.reactions), len(compr_model.reactions)))
    for j in range(subT.shape[1]):
        subT[compr_model.reactions[j].subset_rxns, j] = compr_model.reactions[j].subset_stoich
    # adapt compressed_model name
    return compr_model, subT

def jBigFraction2sympyRat(val): 
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())
    # numer = val.getNumerator()
    # if numer.bitLength() <= 63:
    #     numer = numer.longValue()
    # else:
    #     numer = str(numer.toString())
    # denom = val.getDenominator()
    # if denom.bitLength() <= 63:
    #     denom = denom.longValue()
    # else:
    #     denom = str(denom.toString())
    # # print(numer, denom)
    # return sympy.Rational(numer, denom)

def jBigIntegerPair2sympyRat(numer, denom): 
    if numer.bitLength() <= 63:
        numer = numer.longValue()
    else:
        numer = str(numer.toString())
    if denom.bitLength() <= 63:
        denom = denom.longValue()
    else:
        denom = str(denom.toString())
    # print(numer, denom)
    return sympy.Rational(numer, denom)

# def sympyRat2jBigFraction(val):
#     numer = val.numerator()
#     denom = val.denominator()
#     if numer.bit_length() <= 63 and denom.bit_lenth() <= 63:
#         return BigFraction(numer, denom)
# ...

def sympyRat2jBigIntegerPair(val):
    numer = val.numerator()
    if numer.bit_length() <= 63:
        numer = BigInteger.valueOf(numer)
    else:
        numer = BigInteger(str(numer))
    denom = val.denominator()
    if denom.bit_length() <= 63:
        denom = BigInteger.valueOf(denom)
    else:
        denom = BigInteger(str(denom))
    return (numer, denom)

# %%
import ch.javasoft.smx.ops.Gauss as Gauss
def remove_conservation_relations_sympy(model):
    # does not modify the model, only returns the reduced stoichiometric matrix
    stoich_mat = cobra.util.array.create_stoichiometric_matrix(model, array_type='dok', dtype=numpy.object)
    # stoich_matT = DefaultBigIntegerRationalMatrix(stoich_mat.shape[1], stoich_mat.shape[0]) # transposition
    # for (r, c), v in stoich_mat.items():
    #     if type(v) is not sympy.Rational:
    #         TypeError('Expected rational numbers as coefficients')
    #     n, d = sympyRat2jBigIntegerPair(v)
    #     stoich_matT.setValueAt(c, r, BigFraction(n, d)) # transposition
    stoich_matT = sympyRatMat2jRatMatTransposed(stoich_mat)
    row_map = jpype.JInt[stoich_matT.getRowCount()] # just a placeholder because we don't care about the row permutation here
    col_map = jpype.JInt[:](range(stoich_matT.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(stoich_matT, False, row_map, col_map)
    basic_metabolites = col_map[0:rank]
    reduced = stoich_mat[basic_metabolites, :]
    return reduced, basic_metabolites, stoich_matT

def sympyRatMat2jRatMatTransposed(mat):
    jmat = DefaultBigIntegerRationalMatrix(mat.shape[1], mat.shape[0])
    for (r, c), v in mat.items():
        if type(v) is not sympy.Rational:
            TypeError('Expected rational numbers as coefficients')
        n, d = sympyRat2jBigIntegerPair(v)
        jmat.setValueAt(c, r, BigFraction(n, d))
    return jmat

def sympyRatMat2jRatMat(mat):
    jmat = DefaultBigIntegerRationalMatrix(mat.shape[0], mat.shape[1])
    for (r, c), v in mat.items():
        if type(v) is not sympy.Rational:
            TypeError('Expected rational numbers as coefficients')
        n, d = sympyRat2jBigIntegerPair(v)
        jmat.setValueAt(r, c, BigFraction(n, d))
    return jmat

def jRatMat2sympyRatMat(jmat):
    num_rows = jmat.getRowCount()
    num_cols = jmat.getColumnCount()
    mat = scipy.sparse.dok_matrix((num_rows, num_cols), dtype=numpy.object)
    for r in range(num_rows):
        for c in range(num_cols):
            n = jmat.getBigIntegerNumeratorAt(r, c)
            if not n.equals(BigInteger.ZERO):
                d = jmat.getBigIntegerDenominatorAt(r, c)
                mat[r, c] = jBigIntegerPair2sympyRat(n, d)
    return mat
