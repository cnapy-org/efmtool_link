import efmtool_link.efmtool_intern as efmtool_intern
import numpy
import cobra
import sympy
import scipy.sparse
import jpype
import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import java.math.BigInteger as BigInteger
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import ch.javasoft.math.BigFraction as BigFraction
import ch.javasoft.smx.ops.Gauss as Gauss
import java.util.HashSet
from cobra.core.configuration import Configuration
import time

# %% make a compressed model 
# subT can be used for conversion between its result and the full model
def compress_model(model, remove_rxns=[], tolerance=0.0):
    remove_rxns = [model.reactions.index(r) for r in remove_rxns]
    # the subsets are in the columns of subset_matrix, its rows correspond to the reactions
    subset_matrix = efmtool_intern.compress_rat_efmtool(cobra.util.array.create_stoichiometric_matrix(model, array_type='dok'),
                     [r.reversibility for r in model.reactions], remove_cr=False, tolerance=tolerance,
                     compression_method=efmtool_intern.subset_compression, remove_rxns=remove_rxns)[1]
    compr_model = model.copy()
    config = Configuration()
    del_rxns = numpy.logical_not(numpy.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale all reactions in this subset
            compr_model.reactions[rxn_idx[i]] *= subset_matrix[rxn_idx[i], j]
            if compr_model.reactions[rxn_idx[i]].lower_bound not in (0, config.lower_bound, -float('inf')):
                compr_model.reactions[rxn_idx[i]].lower_bound/= abs(subset_matrix[rxn_idx[i], j])
            if compr_model.reactions[rxn_idx[i]].upper_bound not in (0, config.upper_bound, float('inf')):
                compr_model.reactions[rxn_idx[i]].upper_bound/= abs(subset_matrix[rxn_idx[i], j])
        compr_model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        compr_model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j]
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            compr_model.reactions[rxn_idx[0]] += compr_model.reactions[rxn_idx[i]]
            # the stoichiometries calculated here are less exact than those in rd from compress_rat_efmtool
            # therefore when calulating e.g. conservation relations of the compressed model a non-zero tolerance
            # may be needed to recover the conservation relations
            if compr_model.reactions[rxn_idx[i]].lower_bound > compr_model.reactions[rxn_idx[0]].lower_bound:
                compr_model.reactions[rxn_idx[0]].lower_bound = compr_model.reactions[rxn_idx[i]].lower_bound
            if compr_model.reactions[rxn_idx[i]].upper_bound < compr_model.reactions[rxn_idx[0]].upper_bound:
                compr_model.reactions[rxn_idx[0]].upper_bound = compr_model.reactions[rxn_idx[i]].upper_bound
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

def remove_conservation_relations(model, return_reduced_only=False, tolerance=0):
    stoich_mat = cobra.util.array.create_stoichiometric_matrix(model, array_type='lil')
    basic_metabolites = efmtool_intern.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=tolerance)
    if return_reduced_only:
        reduced = stoich_mat[basic_metabolites, :]
        return reduced, basic_metabolites
    else:
        dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
        print("The following metabolites have been removed from the model:")
        print(dependent_metabolites)
        for m in dependent_metabolites:
            model.metabolites.get_by_id(m).remove_from_model()
        return basic_metabolites

def compress_model_sympy(model, remove_rxns=None, rational_conversion='base10'):
    # modifies the model that is passed as first parameter; if you want to preserve the original model copy it first
    # removes reactions with bounds (0, 0)
    # all irreversible reactions in the compressed model will flow in the forward direction
    # does not remove the conservation relations, see remove_conservation_relations_sympy for this
    if remove_rxns is None:
        remove_rxns = []
    config = Configuration()
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    stoich_mat = DefaultBigIntegerRationalMatrix(num_met, num_reac)
    # reversible = jpype.JBoolean[num_reac]
    reversible = jpype.JBoolean[:]([r.reversibility for r in model.reactions])
    start_time = time.monotonic()
    flipped = []
    for i in range(num_reac):
        if model.reactions[i].bounds == (0, 0): # blocked reaction
            remove_rxns.append(model.reactions[i].id)
        elif model.reactions[i].upper_bound <= 0: # can run in backwards direction only (is and stays classified as irreversible)
            model.reactions[i] *= -1
            flipped.append(i)
            print("Flipped", model.reactions[i].id)
        # have to use _metabolites because metabolites gives only a copy
        for k, v in model.reactions[i]._metabolites.items():
            if type(v) is float or type(v) is int:
                if type(v) is int or v.is_integer():
                    # v = int(v)
                    # n = int2jBigInteger(v)
                    # d = BigInteger.ONE
                    v = sympy.Rational(v) # for simplicity and actually slighlty faster (?)
                else:
                    v = sympy.nsimplify(v, rational=True, rational_conversion=rational_conversion)
                    # v = sympy.Rational(v)
                model.reactions[i]._metabolites[k] = v # only changes coefficient in the model, not in the solver
            elif type(v) is not sympy.Rational:
                raise TypeError
            n, d = sympyRat2jBigIntegerPair(v)
            # does not work although there is a public void setValueAt(int row, int col, BigInteger numerator, BigInteger denominator) method
            # leads to kernel crash directly or later
            # stoic_mat.setValueAt(compr_model.metabolites.index(k.id), i, n, d)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, BigFraction(n, d))
            # reversible[i] = compr_model.reactions[i].reversibility # somehow makes problems with the smc.compress call
    
    smc = StoichMatrixCompressor(efmtool_intern.subset_compression)
    if len(remove_rxns) == 0:
        reacNames = jpype.JString[num_reac]
        remove_rxns = None
    else:
        reacNames = jpype.JString[:](model.reactions.list_attr('id'))
        remove_rxns = java.util.HashSet(remove_rxns) # works because of some jpype magic
        print("Removing", remove_rxns.size(), "reactions:")
        print(remove_rxns.toString())
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, remove_rxns)
    del remove_rxns
    print(time.monotonic() - start_time) # 20 seconds in iJO1366 without remove_rxns
    start_time = time.monotonic()

    # would be faster to do the computations with floats and afterwards substitute the coefficients
    # with rationals from efmtool
    subset_matrix= efmtool_intern.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = numpy.logical_not(numpy.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            factor = jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(rxn_idx[i], j))
            # factor = jBigFraction2intORsympyRat(comprec.post.getBigFractionValueAt(rxn_idx[i], j)) # does not appear to make a speed difference
            model.reactions[rxn_idx[i]] *=  factor #subset_matrix[rxn_idx[i], j]
            # factor = abs(float(factor)) # context manager has trouble with non-float bounds
            if model.reactions[rxn_idx[i]].lower_bound not in (0, config.lower_bound, -float('inf')):
                model.reactions[rxn_idx[i]].lower_bound/= abs(subset_matrix[rxn_idx[i], j]) #factor
            if model.reactions[rxn_idx[i]].upper_bound not in (0, config.upper_bound, float('inf')):
                model.reactions[rxn_idx[i]].upper_bound/= abs(subset_matrix[rxn_idx[i], j]) #factor
        model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j] # use rationals here?
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            model.reactions[rxn_idx[0]] += model.reactions[rxn_idx[i]]
            if model.reactions[rxn_idx[i]].lower_bound > model.reactions[rxn_idx[0]].lower_bound:
                model.reactions[rxn_idx[0]].lower_bound = model.reactions[rxn_idx[i]].lower_bound
            if model.reactions[rxn_idx[i]].upper_bound < model.reactions[rxn_idx[0]].upper_bound:
                model.reactions[rxn_idx[0]].upper_bound = model.reactions[rxn_idx[i]].upper_bound
            del_rxns[rxn_idx[i]] = True
    print(time.monotonic() - start_time) # 11 seconds in iJO1366
    del_rxns = numpy.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = numpy.zeros((num_reac, len(model.reactions)))
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = model.reactions[j].subset_stoich
    for i in flipped: # adapt so that it matches the reaction direction before flipping
            subT[i, :] *= -1
    # adapt compressed_model name
    return subT

def get_rxns_in_subsets(compr_model):
    # build subT from subset_rxns, subset_stoich (probably requires original number of reactions)
    pass

def jBigFraction2sympyRat(val): 
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())

def jBigFraction2intORsympyRat(val): 
    if val.getDenominator().equals(BigInteger.ONE):
        return jBigInteger2int(val.getNumerator())
    else:
        return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())

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

def int2jBigInteger(val):
    if val.bit_length() <= 63:
        return BigInteger.valueOf(val)
    else:
        return BigInteger(str(val))

def jBigInteger2int(val):
    if val.bitLength() <= 63:
        return val.longValue()
    else:
        return int(val.toString())

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

def remove_conservation_relations_sympy(model, return_reduced_only=False):
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
    if return_reduced_only:
        reduced = stoich_mat[basic_metabolites, :]
        return reduced, basic_metabolites, stoich_matT
    else:
        dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
        print("The following metabolites have been removed from the model:")
        print(dependent_metabolites)
        for m in dependent_metabolites:
            model.metabolites.get_by_id(m).remove_from_model()
        return basic_metabolites, stoich_matT

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

# def dokRatMat2dokFloatMat(mat):
#     fmat = scipy.sparse.dok_matrix((mat.shape[0], mat.shape[1]))
#     for (r, c), v in mat.items():
#         fmat[r, c] = float(v)
#     return fmat

def dokRatMat2lilFloatMat(mat):
    fmat = scipy.sparse.lil_matrix((mat.shape[0], mat.shape[1]))
    for (r, c), v in mat.items():
        fmat[r, c] = float(v)
    return fmat
