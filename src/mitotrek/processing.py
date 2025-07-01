import pandas as pd
import numpy as np


def load_single(data_dir, file_prefix, min_cell_conf_detected=3, strand_corr=0.65, vmr=0.01,
	cell_whitelist=None, remove_low_heteroplasmy=True, remove_empty_cells=False):
	"""Load a single sample.

	:param data_dir: Directory containing all variant_stats and cell_heteroplasmic_df files
	:param file_prefix: File name prefix for the aforementioned files
	:param min_cell_conf_detected: minimum threshold for cells confidently detected in  the sample
	:param strand_corr: Minimum threshold for variant strand correlation across cells
	:param vmr: Minimum threshold for variant vmr across cells
	:param remove_low_heteroplasmy: Set anything below 0.05 to 0 since that's roughly limit
	:return: Summary of variants and cell x variant matrix
	"""
	variant_file = data_dir + '{}.variant_stats.tsv.gz'.format(file_prefix)
	variant_stats = pd.read_csv(variant_file, compression='gzip', sep='\t')
	variant_stats = variant_stats.set_index('variant')
	
	valid_variants = variant_stats.loc[variant_stats['n_cells_conf_detected'] >= min_cell_conf_detected, :]
	valid_variants = valid_variants.loc[valid_variants['strand_correlation'] > strand_corr, :]
	valid_variants = valid_variants.loc[valid_variants['vmr'] > vmr, :]
	valid_variants = valid_variants.sort_values('n_cells_conf_detected', ascending=False)

	# disregard variants in a troublesome polymorphic site
	blacklist_vars = [str(y)+'CCCTCCC'[i]+'>' for i, y in enumerate(range(307, 314))]
	valid_variants = valid_variants.drop(valid_variants.index.intersection(blacklist_vars), axis=0)
	
	heteroplasmy_file = data_dir + '{}.cell_heteroplasmic_df.tsv.gz'.format(file_prefix)
	hetero_df = pd.read_csv(heteroplasmy_file, compression='gzip', sep='\t', index_col=0).fillna(0)
	hetero_df = hetero_df[valid_variants.index]
	hetero_df.index = ['{}#{}'.format(file_prefix, bc) for bc in hetero_df.index]

	print('{}: {} cells, {} variants.'.format(file_prefix, hetero_df.shape[0], hetero_df.shape[1]))

	if remove_low_heteroplasmy:
		hetero_df[hetero_df < 0.05] = 0

	# remove cells without any variant
	if remove_empty_cells:
		hetero_df =	hetero_df.loc[hetero_df.sum(axis=1) > 0, :]

	# keep only cells in whitelist if provided
	if cell_whitelist is not None:
		hetero_df = hetero_df.loc[hetero_df.index.intersection(cell_whitelist), :]

	return valid_variants, hetero_df


def load_multiple(data_dir, file_prefixes, sample_names=False,
	min_cell_conf_detected=3, strand_corr=0.65, vmr=0.01, 
	cell_whitelist=None, remove_low_heteroplasmy=True, remove_empty_cells=False):
	"""Load multiple samples and merge them all into one cell x variant matrix.

	:param data_dir: Directory containing all variant_stats and cell_heteroplasmic_df files
	:param file_prefixes: File name prefixes for the aforementioned files
	:param sample_names: A list of names if don't want to use file prefixes
	:param min_cell_conf_detected: minimum threshold for cells confidently detected in  the sample
	:param strand_corr: Minimum threshold for variant strand correlation across cells
	:param vmr: Minimum threshold for variant vmr across cells
	:param remove_low_heteroplasmy: Set anything below 0.05 to 0 since that's roughly limit
	:return: Summary of variants and cell x variant matrix
	"""
	if sample_names is False:
		# option to rename samples different from file names
		sample_names = file_prefixes


	variant_sum, hetero_dict = [], dict()
	for i, cur_file_prefix in enumerate(file_prefixes):
		cur_sample = sample_names[i]

		variant_file = data_dir + '{}.variant_stats.tsv.gz'.format(cur_file_prefix)
		variant_stats = pd.read_csv(variant_file, compression='gzip', sep='\t')
		variant_stats = variant_stats.set_index('variant')
		variant_stats['sample'] = cur_sample
		
		valid_variants = variant_stats.loc[variant_stats['n_cells_conf_detected'] >= min_cell_conf_detected , :]
		valid_variants = valid_variants.sort_values('n_cells_conf_detected', ascending=False)
		
		heteroplasmy_file = data_dir + '{}.cell_heteroplasmic_df.tsv.gz'.format(cur_file_prefix)
		hetero_df = pd.read_csv(heteroplasmy_file, compression='gzip', sep='\t', index_col=0).fillna(0)
		hetero_df = hetero_df[valid_variants.index]
		hetero_df.index = ['{}#{}'.format(cur_sample, bc) for bc in hetero_df.index]
		
		variant_sum.append(valid_variants)
		hetero_dict[cur_sample] = hetero_df

	variant_sum = pd.concat(variant_sum).sort_values('n_cells_conf_detected', ascending=False)

	# union method: keep a variant in all samples if it satisfies criteria in any sample
	filter_cond = (variant_sum['strand_correlation'] > strand_corr) & (variant_sum['vmr'] > vmr)
	select_vars = filter_cond[filter_cond].index.unique().tolist()

	# disregard variants in a troublesome polymorphic site
	blacklist_vars = [str(y)+'CCCTCCC'[i]+'>' for i, y in enumerate(range(307, 314))]
	select_vars = [x for x in select_vars if not any(y in x for y in blacklist_vars)]
	
	print("{} variants detected.".format(len(select_vars)))

	variant_sum = variant_sum.loc[select_vars, :].sort_values('n_cells_conf_detected', ascending=False)
	for key in hetero_dict.keys():
		hetero_dict[key] = hetero_dict[key][hetero_dict[key].columns.intersection((pd.Index(select_vars)))]
		hetero_dict[key] = hetero_dict[key].loc[hetero_dict[key].sum(axis=1) > 0, :]
		print('{}: {} cells, {} variants.'.format(key, hetero_dict[key].shape[0], hetero_dict[key].shape[1]))


	all_hetero_df = pd.concat([hetero_dict[x] for x in hetero_dict.keys()]).fillna(0)
	if remove_low_heteroplasmy:
		all_hetero_df[all_hetero_df < 0.05] = 0

	# remove cells without any variant
	if remove_empty_cells:
		all_hetero_df =	all_hetero_df.loc[all_hetero_df.sum(axis=1) > 0, :]

	# keep only cells in whitelist if provided
	if cell_whitelist is not None:
		all_hetero_df = all_hetero_df.loc[all_hetero_df.index.intersection(cell_whitelist), :]

	return variant_sum, all_hetero_df
