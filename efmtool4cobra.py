import efmtool_link
import numpy
import cobra

# %% make a compressed model 
# subT can be used for conversion between its result and the full model
def compress_model(model):
    # the subsets are in the columns of subset_matrix, its rows correspond to the reactions
    subset_matrix = efmtool_link.compress_rat_efmtool(cobra.util.array.create_stoichiometric_matrix(model, array_type='dok'),
                     [r.reversibility for r in model.reactions], remove_cr=False, compression_method=efmtool_link.subset_compression)[1]
    compressed_model = model.copy()
    # del_rxns = [] #numpy.zeros(subT.shape[0], dtype=numpy.bool)
    del_rxns = numpy.logical_not(numpy.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        # print(rxn_idx)
        for i in range(len(rxn_idx)): # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            compressed_model.reactions[rxn_idx[i]] *= subset_matrix[rxn_idx[i], j]
        compressed_model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        compressed_model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j]
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            # fix so that at least the reversibility is properly constrained
            compressed_model.reactions[rxn_idx[0]] += compressed_model.reactions[rxn_idx[i]]
            if compressed_model.reactions[rxn_idx[i]].lower_bound == 0:
                # print(rxn_idx[0])
                compressed_model.reactions[rxn_idx[0]].lower_bound = 0
            del_rxns[rxn_idx[i]] = True
            # del_rxns.append(compressed_model.reactions[rxn_idx[i]].id)
    del_rxns = numpy.where(del_rxns)[0]
    # for i in range(len(del_rxns)):
    #     compressed_model.reactions.get_by_id(del_rxns[i]).remove_from_model(remove_orphans=True)
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        compressed_model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    # does not remove the conservation relations
    subT = numpy.zeros((len(model.reactions), len(compressed_model.reactions)))
    for j in range(subT.shape[1]):
        subT[compressed_model.reactions[j].subset_rxns, j] = compressed_model.reactions[j].subset_stoich
    # adapt compressed_model name
    return compressed_model, subT
