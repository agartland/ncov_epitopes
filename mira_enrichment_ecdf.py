import pandas as pd
import numpy as np
from os.path import join as opj
import sys
from functools import partial
import dill
from glob import glob

import scipy.cluster.hierarchy as sch

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.distributions.empirical_distribution import ECDF

from fg_shared import *

import pwseqdist as pwsd

sys.path.append(opj(_git, 'tcrdist3'))
import tcrdist as td
from tcrdist.repertoire import TCRrep
from tcrdist.pgen import OlgaModel

sys.path.append(opj(_git, 'utils'))
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from ecdf_helpers import *

def _pwrect(tr, clone_df1, clone_df2=None, metric='tcrdist', chain='beta', ncpus=3):
    if clone_df2 is None:
        clone_df2 = tr.clone_df
    
    col = f'cdr3_{chain[0]}_aa'

    if metric == 'tcrdist':
        tr.cpus = ncpus
        tr.chains = [chain]
        tr.compute_rect_distances(df=clone_df1, df2=clone_df2, store=False)
        pwmat = getattr(tr, f'rw_{chain}')
    elif metric == 'tcrdist-cdr3':
        pwmat = pwsd.apply_pairwise_rect(metric=pwsd.metrics.nb_vector_tcrdist,
                                         seqs1=clone_df1[col],
                                         seqs2=clone_df2[col],
                                         ncpus=ncpus,
                                         use_numba=True)
    elif metric == 'edit':
        pwmat = pwsd.apply_pairwise_rect(metric=pwsd.metrics.nb_vector_editdistance,
                                         seqs1=clone_df1[col],
                                         seqs2=clone_df2[col],
                                         ncpus=ncpus,
                                         use_numba=True)
    return pwmat

out_folder = opj(_fg_data, 'ncov_tcrs', 'adaptive_bio_r2', 'ecdf_2020-AUG-31')

"""Load the reference sequences"""
ref_fn = opj(_fg_data, 'tcrdist/datasets', 'human_T_beta_bitanova_unique_clones_sampled_1220K.csv')
# olga_ref_fn = opj(_fg_data, 'tcrdist/datasets', 'human_T_beta_sim1000K.csv')

ref_df = pd.read_csv(ref_fn)
ref_df.columns = [{'v_b_name':'v_b_gene',
                   'j_b_name':'j_b_gene',
                   'cdr3_b_aa':'cdr3_b_aa'}.get(c, c) for c in ref_df.columns]
ref_df.loc[:, 'count'] = 1
ref_tr = TCRrep(cell_df=ref_df.iloc[:100], 
                organism='human', 
                chains=['beta'],
                compute_distances=False)

rep_filenames = glob(opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/pw_computed/*.dill'))

rep_filenames = ['mira_epitope_61_428_GRLQSLQTY_LITGRLQSL_RLQSLQTYV.tcrdist3.csv.dill',
                 'mira_epitope_131_104_HLMSFPQSA_YHLMSFPQSA.tcrdist3.csv.dill',
                 'mira_epitope_60_436_MWSFNPETNI_SFNPETNIL_SMWSFNPET.tcrdist3.csv.dill']

for rep_fn in rep_filenames:
    for metric, fclust_thresh in ['tcrdist', 'tcrdist-cdr3', 'edit']:
        if 'tcr' in metric:
            metric_thresholds = np.arange(1, 101)
            fcluster_thresholds = [0, 25, 50]
        else:
            metric_thresholds = np.arange(1, 9)
            fcluster_thresholds = [0, 1, 2]

        epitope_name = os.path.split(rep_filenames[0])[1].split('.')[0]
        rep_fn = opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/pw_computed', rep_fn)

        with open(rep_fn, 'rb') as fh:
            tr = dill.load(fh)

        """Compute repertoire PW distances and create flat clusters"""
        rep_pwmat = _pwrect(tr, clone_df1=tr.clone_df, metric=metric)

        ref_pwmat = _pwrect(tr, clone_df1=tr.clone_df,
                                clone_df2=ref_tr.clone_df, metric=metric)

        for fclust_thresh in fcluster_thresholds:
            if fclust_thresh > 0:
                Z = sch.linkage(rep_pwmat, method='complete')
                labels = sch.fcluster(Z, t=fclust_thresh, criterion='distance')
            else:
                labels = np.arange(1, rep_pwmat.shape[0] + 1)

            """Compute ECDF for each cluster within the repertoire"""
            rep_ecdf = np.zeros((int(np.max(labels)), len(metric_thresholds)))
            for lab in range(1, np.max(labels) + 1):
                lab_ind = labels == lab
                rep_ecdf[lab-1, :] = compute_ecdf(np.mean(rep_pwmat[lab_ind, :], axis=0), thresholds=metric_thresholds)

            """Compute distances to the reference for each cluster and compute ECDF vs reference"""
            ref_ecdf = np.zeros((int(np.max(labels)), len(metric_thresholds)))
            for lab in range(1, np.max(labels) + 1):
                lab_ind = labels == lab
                ref_ecdf[lab-1, :] = compute_ecdf(np.mean(ref_pwmat[lab_ind, :], axis=0), thresholds=metric_thresholds)

            with(open(opj(out_folder, f'{epitope_name}_{metric}_repclust_FCT{fclust_thresh}.npz', 'wb'))) as fh:
                np.savez(fh, rep_clust_ecdf, allow_pickle=False)



"""Plotting"""
    """Compute pgen of each MIRA TCR"""
    olga_beta  = OlgaModel(chain_folder="human_T_beta", recomb_type="VDJ")
    tr.clone_df['pgen_cdr3_b_aa'] = olga_beta.compute_aa_cdr3_pgens(tr.clone_df.cdr3_b_aa)
    tr.clone_df = tr.clone_df.assign(pgen_b_color=tr.clone_df['pgen_cdr3_b_aa'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)))

    """Force pgen > 0: there were 7 CDR3 alphas with pgen = 0"""
    # tr.clone_df = tr.clone_df.loc[(tr.clone_df['pgen_cdr3_b_aa'] > 0)]