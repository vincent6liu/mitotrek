import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact
from joblib import Parallel, delayed


def assign_cell_to_clones(heteroplasmy_df, max_abundance=0.2, min_abundance=3, bin_cutoff=0.1, corr_cutoff=0.5):
    """Call clones from heteroplasmy data.
    Disregard variants detected in more than default 5% of cells (technical or not informative)
    Disregard variants detected in less than default 3 cells (to speed up computation)
    Compute variant-variant correlation and group correlated variants as one clone barcode
    Define a variant as detected in a cell if heteroplasmy >= 10%
    :param heteroplasmy_df: Cell x variant matrix.
    :param max_abundance: max proportion of cells a variant can be detected
    :param min_abundance: min number of cells a variant can be detected
    :param corr_cutoff: variant-variant correlation cutoff to group variants
    :return: a three tuple where first element is pd.Series of cell-clone assignment
        second element is dictionary denoting variant(s) defining each clone
        third element is dictionary including cells assigned to multiple clones
    """
    n_cells = heteroplasmy_df.shape[0]
    bin_hetero_df = heteroplasmy_df >= bin_cutoff
    var_cell_count = bin_hetero_df.sum()
    
    print('{} cells in heteroplasmy df.'.format(n_cells))
    
    # drop variants present in more than max_abundance of cells
    var_discard = var_cell_count[var_cell_count > n_cells*max_abundance].index
    if len(var_discard) > 0:
        heteroplasmy_df = heteroplasmy_df.drop(var_discard, axis=1)
        bin_hetero_df = bin_hetero_df.drop(var_discard, axis=1)
    
    # drop variants present in less than min_abundance cells
    var_discard = var_cell_count[var_cell_count < min_abundance].index
    if len(var_discard) > 0:
        heteroplasmy_df = heteroplasmy_df.drop(var_discard, axis=1)
        bin_hetero_df = bin_hetero_df.drop(var_discard, axis=1)
    
    available_cells_to_assign = n_cells - bin_hetero_df.sum(axis=1).value_counts()[0]
    print('{} cells available to be assigned.'.format(available_cells_to_assign))
    print('{} variants used to call clones...'.format(heteroplasmy_df.shape[1]))
    
    # compute pearson correlation btw variants using binary <= or > 0.1 heteroplasmy
    # rationale here is the exact heteroplasmy level not trustworthy due to 
    # random mito distribution after cell division
    var_corr_df = bin_hetero_df.corr()
    
    # correlation cutoff to determine if two variants are linked
    var_corr_bool_df = var_corr_df > corr_cutoff
    
    # construct a graph where nodes are variants and edges are added if correlation > threshold
    edges = []
    for cur_var in var_corr_bool_df.columns:
        var_corr_bool_col = var_corr_bool_df[cur_var]
        corr_var_list = var_corr_bool_col[var_corr_bool_col].index.tolist()
        corr_var_list = list(set(corr_var_list) - set([cur_var]))
        for corr_var in corr_var_list:
            edges.append((corr_var, cur_var))
    G = nx.Graph()
    G.add_nodes_from(var_corr_bool_col.index.tolist())
    G.add_edges_from(edges)

    # break down the graph to pull out correlated variants
    graph_connected_components = [x for x in nx.connected_components(G)]
    var_uncorr = [x.pop() for x in graph_connected_components if len(x) == 1]
    var_clusters = [list(x) for x in graph_connected_components if len(x) > 1]  # correlated variants
    var_clusters_dict = dict(zip(['var_cluster_{}'.format(x) for x in range(1, len(var_clusters)+1)], var_clusters))
    
    # assign cells to clones defined by a single variant
    clone_marker_dict = dict()  # record clone-variant correspondence
    clone_assign_dict = defaultdict(list)  # record all clone assignment for cells
    for i, cur_var in enumerate(var_uncorr):
        cur_clone = 'clone_{}'.format(i+1)
        clone_marker_dict[cur_clone] = [cur_var]
        cur_pos_cells = bin_hetero_df.index[bin_hetero_df[cur_var] == 1]
        for cur_cell in cur_pos_cells:
            clone_assign_dict[cur_cell].append(cur_clone)
    
    # assign cells to clones defined by multiple correlated variants
    for j, cur_var_cluster in enumerate(var_clusters):
        cur_clone = 'clone_{}'.format(i+j+1)
        clone_marker_dict[cur_clone] = list(cur_var_cluster)
        # require a cell to have ALL correlated varaints to be assigned to the clone
        cur_pos_cells = bin_hetero_df.index[bin_hetero_df[cur_var_cluster].sum(axis=1) == len(cur_var_cluster)]
        for cur_cell in cur_pos_cells:
            clone_assign_dict[cur_cell].append(cur_clone)
    
    # pd.Series of cells uniquely assigned and their corresponding clone
    # we omit cells that aren't assigned or multi-assigned to be conservative at the cost of sensitivity
    uniquely_assinged_cells = [(x, y[0]) for x, y in clone_assign_dict.items() if len(y)==1]
    uniquely_assinged_cells = pd.DataFrame(uniquely_assinged_cells, columns=['cell', 'clone'])
    uniquely_assinged_cells = uniquely_assinged_cells.set_index('cell')['clone']
    
    n_multi_assigned = sum([1 for y in clone_assign_dict.values() if len(y)>1])
    
    print('{} cells assigned to more than one clone and discarded.'.format(n_multi_assigned))
    print('{} clones defined by more than one variant.'.format(len(var_clusters)))
    print('{} cells assigned to {} clones.'.format(uniquely_assinged_cells.shape[0], 
                                                   len(uniquely_assinged_cells.unique())))
    
    return (uniquely_assinged_cells, clone_marker_dict, clone_assign_dict)


def rank_cluster_specific_variants(heteroplasmy_df, binary_cell_label, hetero_threshold=0.1, ncores=-1, 
    min_cell_number=6, flag_control_cell_number=5, keep_negative_enrichment=False, transition_only=True, high_freq_filter=True):
    """Detect variants specific to a group of cells defined by binary_cell_label.
    Binarize the heteroplasmy df and run fisher's exact test for each variant.
    Note the function automatically takes the intersection of cells between heteroplasmy_df and binary_cell_label.

    :param heteroplasmy_df: Cell x variant matrix.
    :param binary_cell_label: Binary pd.Series with 1 denoting cells in group of interest and 0 comparison group
    :param hetero_threshold: Heteroplasmy threshold to use for binarization
    :param ncores: Number of cores to use for parallelized fisher's exact test across variants.
    :return: pd.DataFrame with odds ratio and p-val for all enriched variants (odds ratio > 1)
    """
    if ncores == -1:
        ncores = multiprocessing.cpu_count()
    
    binary_cell_label = binary_cell_label[binary_cell_label.index.intersection(heteroplasmy_df.index)]
    print('{} cells in target group.'.format(binary_cell_label.sum()))
    print('{} cells in control group.'.format((~binary_cell_label).sum()))

    bin_hetero_df = (heteroplasmy_df > hetero_threshold).astype(int)
    bin_hetero_df = bin_hetero_df.loc[binary_cell_label.index, :]
    bin_hetero_df = bin_hetero_df.loc[:, bin_hetero_df.sum() > 0]
    
    if high_freq_filter:
        # keep only variants present in less than 10% of all cells, otherwise odds ratio becomes unreliable
        frac_group = bin_hetero_df.sum() / bin_hetero_df.shape[0]
        frac_group = frac_group < 0.1
        bin_hetero_df = bin_hetero_df.loc[:, frac_group]

    # keep only variants present in at least min_cell_number of cells
    bin_hetero_df = bin_hetero_df.loc[:, bin_hetero_df.sum() >= min_cell_number]
    
    all_vars = bin_hetero_df.columns
    fisher_results = Parallel(n_jobs=ncores)(delayed(fisher_exact)(pd.crosstab(binary_cell_label, bin_hetero_df[var])) for var in all_vars)
    fisher_results = pd.DataFrame(fisher_results, index=bin_hetero_df.columns, columns=['odds_ratio', 'p_val'])
    fisher_results['target_count'] = bin_hetero_df.loc[binary_cell_label, fisher_results.index].sum()
    fisher_results['control_count'] = bin_hetero_df.loc[~binary_cell_label, fisher_results.index].sum()
    fisher_results = fisher_results.sort_values(['p_val'])

    # flag variants in too many control set
    fisher_results['flag'] = fisher_results['control_count'] > flag_control_cell_number
    
    # whether to keep variants with negative enrichment
    if not keep_negative_enrichment:
        fisher_results = fisher_results.loc[fisher_results['odds_ratio'] > 1, :]

    # keep only transition mutations, otherwise much more likely to be technical
    if transition_only:
        transition_mutations = ['A>G', 'G>A', 'T>C', 'C>T']
        var_use = pd.Index([x for x in fisher_results.index if any([y in x for y in transition_mutations])])
        fisher_results = fisher_results.loc[var_use, :]
    
    return fisher_results


def compute_affinity(heteroplasmy_df, dist='weighted_jaccard'):
    """Compute affinity matrix using cell heteroplasmy df.

    :param heteroplasmy_df: Cell x variant matrix.
    :return: Cell-cell affinity_df.
    """
    if dist == 'weighted_jaccard':
        dist = _weighted_jaccard

    affinity_df = squareform(pdist(heteroplasmy_df, dist))
    affinity_df = pd.DataFrame(affinity_df, index=heteroplasmy_df.index, columns=heteroplasmy_df.index)

    return affinity_df


def compute_cluster_clonality_index(affinity_df, cluster_assignment, scale_factor=10, round_digits=4):
    """Quantify how clonal each cluster of cells is.
    Note the function automatically takes the intersection of cells between affinity_df and cluster_assignment.

    :param affinity_df: Cell-cell affinity_df.
    :param cluster_assignment: pd.Series encoding cluster assignment for cells.
    :return: A pd.Series encoding cluster clonality index for all clusters.
    """
    cluster_clonality_index = dict()
    for i in sorted(cluster_assignment.unique()):
        use_index = cluster_assignment[cluster_assignment==i].index.intersection(affinity_df.index)
        cluster_clonality_index[i] = affinity_df.loc[use_index, use_index].sum().sum()  # sum of all edges within cluster
        cluster_clonality_index[i] /= len(use_index)**2  # normalize wrt number of nodes in the cluster
    
    cluster_clonality_index = (pd.Series(cluster_clonality_index) * scale_factor).round(round_digits)   # scale up
    cluster_clonality_index = cluster_clonality_index.sort_values(ascending=False)

    return cluster_clonality_index


def computer_intercluster_connectedness(affinity_df, cluster_assignment):
    return


def clone_calling_accuracy(true_labels, called_labels):
    concat_labels = pd.concat([pd.Series(true_labels), pd.Series(called_labels)], axis=1)
    concat_labels.columns = ['true', 'called']
    unique_true_clones = concat_labels['true'].unique()
    unique_called_clones = concat_labels['called'].unique()

    clone_map = dict()
    for tc in unique_true_clones:
        cur_mfc = concat_labels.loc[concat_labels['true']==tc, 'called'].mode()[0]
        cur_true_for_mfc = concat_labels.loc[concat_labels['called']==cur_mfc, 'true'].mode()[0]
        if cur_true_for_mfc == tc:
            clone_map[tc] = cur_mfc

    correct_count = sum([1 for i, x in enumerate(true_labels) if x in clone_map and called_labels[i]==clone_map[x]])
    accuracy = correct_count / len(true_labels)

    return accuracy


def _weighted_jaccard(a, b):
    stacked = np.array([a, b])
    return (stacked.min(axis=0).sum() / stacked.max(axis=0).sum())
