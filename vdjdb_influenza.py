import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython
import json

from fg_shared import *
import pwseqdist
import fishersapi
from palmotif import compute_pal_motif, svg_logo
from hierdiff import plot_hclust_props

import tcrdist as td
from tcrdist.mappers import vdjdb_to_tcrdist2, vdjdb_to_tcrdist2_mapping_TRA, vdjdb_to_tcrdist2_mapping_TRB
from tcrdist.repertoire import TCRrep
from tcrdist.plotting import plot_pairings
from tcrsampler.sampler import TCRsampler
from tcrdist.adpt_funcs import get_centroid_seq, get_centroid_seq_alpha
from tcrdist.summarize import _select
from tcrdist.rep_diff import hcluster_diff, member_summ
from tcrdist.tree import TCRtree

sys.path.append(opj(_git, 'utils'))
from pngpdf import PngPdfPages

proj_folder = opj(_fg_data, 'ncov_epitopes', 'data', 'vdjdb_examples')

raw = pd.read_csv(opj(proj_folder, 'influenza_2020-SEP.tsv'), sep = '\t')

pmids = ['PMID:25609818', 'PMID:28423320']

selin = raw.loc[raw['Reference'] == pmids[0]]
selin = selin.assign(cohort=selin['Meta'].map(lambda d: json.loads(d)['subject.cohort']),
                     subject=selin['Meta'].map(lambda d: json.loads(d)['subject.id']))

selin_a = selin.loc[selin['Gene'] == 'TRA'].rename(vdjdb_to_tcrdist2_mapping_TRA, axis=1)
selin_b = selin.loc[selin['Gene'] == 'TRB'].rename(vdjdb_to_tcrdist2_mapping_TRB, axis=1)

trb = TCRrep(cell_df=selin_b,
            organism='human',
            chains=['beta'])

tra = TCRrep(cell_df=selin_a,
            organism='human',
            chains=['alpha'])

svg = plot_pairings(cell_df=trb.clone_df,
                                cols = ['v_b_gene', 'j_b_gene'],
                                count_col='count',
                                other_frequency_threshold = 0.01)
IPython.display.SVG(data=svg)

svg = plot_pairings(cell_df=tra.clone_df,
                                cols = ['v_a_gene', 'j_a_gene'],
                                count_col='count',
                                other_frequency_threshold = 0.01)
IPython.display.SVG(data=svg)

#t = TCRsampler()
# t.download_background_file('ruggiero_mouse_sampler.zip')
#tcrsampler_beta = TCRsampler(default_background='ruggiero_mouse_beta_t.tsv.sampler.tsv')
#tcrsampler_alpha = TCRsampler(default_background='ruggiero_mouse_alpha_t.tsv.sampler.tsv')

tcrtree = TCRtree(tcrrep=tra, html_name=opj(proj_folder, 'Gil_alpha_hierdiff.html'))
tcrtree.default_hcluster_diff_kwargs['x_cols'] = ['cohort']
tcrtree.default_plot_hclust_props['alpha'] = 0.05
tcrtree.default_plot_hclust_props['tooltip_cols'].append('ref_size_olga_alpha')
tcrtree.default_plot_hclust_props['tooltip_cols'].append('ref_unique_olga_alpha')
tcrtree.default_plot_hclust_props['tooltip_cols'].append('percent_missing_olga_alpha')
tcrtree.build_tree()

tcrtree = TCRtree(tcrrep=trb, html_name=opj(proj_folder, 'Gil_beta_hierdiff.html'))
tcrtree.default_hcluster_diff_kwargs['x_cols'] = ['cohort']
tcrtree.default_plot_hclust_props['alpha'] = 0.05
tcrtree.default_plot_hclust_props['tooltip_cols'].append('ref_size_olga_beta')
tcrtree.default_plot_hclust_props['tooltip_cols'].append('ref_unique_olga_beta')
tcrtree.default_plot_hclust_props['tooltip_cols'].append('percent_missing_olga_beta')
tcrtree.build_tree()


"""Cluster CID975 shows how there can be a large public cluster with a strong motif,
yet all the cluster members are private"""
"""CID 975 (i = 463)
36 / 61 are private sequences
12 / 16 individuals
"""
print(tra.clone_df.loc[tra.hcluster_df_detailed.loc[463, 'neighbors']].groupby('cdr3_a_aa')['subject'].count().sort_values())
print(tra.clone_df.loc[tra.hcluster_df_detailed.loc[463, 'neighbors']].groupby('cdr3_a_aa')['subject'].count().sort_values().value_counts())

"""CID 934 (i = 422)
28 / 92 are private sequences
12 / 16 individuals
"""
print(tra.clone_df.loc[tra.hcluster_df_detailed.loc[422, 'neighbors']].groupby('cdr3_a_aa')['subject'].count().sort_values())
print(tra.clone_df.loc[tra.hcluster_df_detailed.loc[422, 'neighbors']].groupby('cdr3_a_aa')['subject'].count().sort_values().value_counts())