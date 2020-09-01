import pandas as pd
import numpy as np
from os.path import join as opj
import sys
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

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

"""TODO:
 - Look at these ratio plots for all the target sequences and overlay with alpha. Any pattern?
 - Summarize for an epitope repertoire and then compute for different repertoires. Any patterns?
 - Maybe these should be differences as opposed to ratios?
 - Show horizontal line for LD = 1, 2, 3 ratio on the TCR plot. How do these compare to the full tcrdist
   distribution from 0 to 100?
 - Are there differences in this relationship by epitope or by sequence within an epitope?"""

def _compute_ecdf(data, counts=None, thresholds=None, n=1):
    """Computes the empirical cumulative distribution function at pre-specified
    thresholds. Assumes thresholds is sorted and should be unique."""
    if thresholds is None:
        thresholds = np.unique(data[:])
    if counts is None:
        counts = np.ones(data.shape)
    
    tot = np.sum(counts)
    # ecdf = np.array([np.sum((data <= t) * counts)/tot for t in thresholds])
    
    """Vectorized and faster, using broadcasting for the <= expression"""
    ecdf = (np.sum((data[:, None] <= thresholds[None, :]) * counts[:, None], axis=0) + 1) / (tot + 1)
    n_ecdf = (np.sum((data[:, None] <= thresholds[None, :]) * counts[:, None], axis=0) >= n).astype(int)
    return ecdf, n_ecdf

"""
from tcrdist.setup_tests import download_and_extract_zip_file
datasets = ['dash.zip',
            'human_T_alpha_beta_sim200K.zip',
            'vdjDB_PMID28636592.zip',
            'sant.csv.zip',
            'bulk.csv.zip',
            'wiraninha_sampler.zip',
            'ruggiero_mouse_sampler.zip',
            'ruggiero_human_sampler.zip',
            'britanova_human_beta_t_cb.tsv.sampler.tsv.zip',
            'emerson_human_beta_t_cmvneg.tsv.sampler.tsv.zip',
            'ruggiero_human_alpha_t.tsv.sampler.tsv.zip',
            'ruggiero_human_beta_t.tsv.sampler.tsv.zip']

for fn in datasets:
    try:
        download_and_extract_zip_file(fn, source='dropbox', dest=opj(_fg_data, 'tcrdist', 'datasets'))
    except KeyError:
        print(fn)
"""

# em = pd.read_csv(opj(_fg_data, 'tcrdist', 'datasets', 'emerson_human_beta_t_cmvneg.tsv.sampler.tsv'), sep = "\t")
em = pd.read_csv(opj(_fg_data, 'tcrdist', 'datasets', 'britanova_human_beta_t_cb.tsv.sampler.tsv'), sep = "\t")
em_ss = em.sample(n=1000000, replace=False, random_state=110820)
em_ss.columns = [{'v_reps':'v_b_gene',
                   'j_reps':'j_b_gene',
                   'cdr3':'cdr3_b_aa'}.get(c, c) for c in em_ss.columns]
em_tr = TCRrep(cell_df=em_ss, 
                organism='human', 
                chains=['beta'],
                compute_distances=False)

dash_fn = opj(_fg_data, 'tcrdist', 'datasets', 'dash_human.csv')

df = pd.read_csv(dash_fn)
tr = TCRrep(cell_df=df, 
            organism='human', 
            chains=['alpha', 'beta'],
            compute_distances=False)

"""Compute pgen of each epitope-specific sequence"""
olga_beta  = OlgaModel(chain_folder = "human_T_beta", recomb_type="VDJ")
olga_alpha = OlgaModel(chain_folder = "human_T_alpha", recomb_type="VJ")

tr.clone_df['pgen_cdr3_b_aa'] = olga_beta.compute_aa_cdr3_pgens(tr.clone_df.cdr3_b_aa)
tr.clone_df['pgen_cdr3_a_aa'] = olga_alpha.compute_aa_cdr3_pgens(tr.clone_df.cdr3_a_aa)

"""Force pgen > 0: there were 7 CDR3 alphas with pgen = 0"""
tr.clone_df = tr.clone_df.loc[(tr.clone_df['pgen_cdr3_a_aa'] > 0) & (tr.clone_df['pgen_cdr3_b_aa'] > 0)]

norm_pgen = mpl.colors.LogNorm(vmin=1e-10, vmax=1e-6) 
norm_a = mpl.colors.LogNorm(vmin=tr.clone_df['pgen_cdr3_a_aa'].min(),
                            vmax=tr.clone_df['pgen_cdr3_a_aa'].max())

norm_b = mpl.colors.LogNorm(vmin=tr.clone_df['pgen_cdr3_b_aa'].min(),
                            vmax=tr.clone_df['pgen_cdr3_b_aa'].max())
def color_lu(norm, colors, pgen):
    i = int(np.floor(norm(pgen) * len(colors)))
    if i >= len(colors):
        i = len(colors) - 1
    if i < 0:
        i = 0
    return tuple(colors[i])

mpl.rcParams['font.size'] = 14

def rect_ecdf(tr, clone_df1, clone_df2, metric, thresholds, ncpus=1, necdf=1, chain='beta'):
    if metric == 'tcrdist':
        tr.cpus = ncpus
        tr.chains = [chain]
        tr.compute_rect_distances(df=clone_df1, df2=clone_df2, store=False)
        pwmat = getattr(tr, f'rw_{chain}')
    elif metric == 'tcrdist-cdr3':
        if chain == 'beta':
            col = 'cdr3_b_aa'
        else:
            col = 'cdr3_a_aa'
        pwmat = pwsd.apply_pairwise_rect(metric=pwsd.metrics.nb_vector_tcrdist,
                                            seqs1=clone_df1[col],
                                            seqs2=clone_df2[col],
                                            ncpus=ncpus,
                                            use_numba=True)
    else:
        if chain == 'beta':
            col = 'cdr3_b_aa'
        else:
            col = 'cdr3_a_aa'
        pwmat = pwsd.apply_pairwise_rect(metric=pwsd.metrics.nb_vector_editdistance,
                                            seqs1=clone_df1[col],
                                            seqs2=clone_df2[col],
                                            ncpus=ncpus,
                                            use_numba=True)
    ecdf = np.zeros((pwmat.shape[0], thresholds.shape[0]))
    n_ecdf = np.zeros((pwmat.shape[0], thresholds.shape[0]))
    for tari in range(pwmat.shape[0]):
        ecdf[tari, :], n_ecdf[tari, :] = _compute_ecdf(pwmat[tari, :], thresholds=thresholds, n=necdf)

    return pwmat, ecdf, n_ecdf

tr.clone_df = tr.clone_df.assign(pgen_b_color=tr.clone_df['pgen_cdr3_b_aa'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)),
                                 pgen_a_color=tr.clone_df['pgen_cdr3_a_aa'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)))

ld_thresh = np.arange(15)
tcr_thresh = np.arange(150)

ld_lim = (0, 13)
tcr_lim = (0, 156) # factor of 12

with PngPdfPages(opj(_fg_data, 'tcrdist', 'figures', '2020-AUG-19', 'dash_contamination_full-tcrdist.pdf')) as pdf:
    for epitope in ['BMLF', 'M1', 'pp65']:
        #epitope = 'M1'

        """Compute the "square" pairwise distances of the epitope-specific sequences"""
        tarmat_ld, tar_ld_ecdf, tar_ld_singl = rect_ecdf(tr, tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               metric='LD',
                                               thresholds=ld_thresh,
                                               ncpus=1, chain='beta', necdf=2)

        tarmat_tcr, tar_tcr_ecdf, tar_tcr_singl = rect_ecdf(tr, tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               metric='tcrdist',
                                               thresholds=tcr_thresh,
                                               ncpus=1, chain='beta', necdf=2)

        bulkmat_ld, bulk_ld_ecdf, bulk_ld_singl = rect_ecdf(tr, tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               em_tr.clone_df,
                                               metric='LD',
                                               thresholds=ld_thresh,
                                               ncpus=3, chain='beta', necdf=1)
        bulkmat_tcr, bulk_tcr_ecdf, bulk_tcr_singl = rect_ecdf(tr, tr.clone_df[tr.clone_df['epitope'] == epitope],
                                               em_tr.clone_df,
                                               metric='tcrdist',
                                               thresholds=tcr_thresh,
                                               ncpus=3, chain='beta', necdf=1)

        """Plot the ECDF for each epitope-specific clone, colored by pgen
        for distances to all other TCRs either in the epitope-specific or bulk data"""
        alpha = 0.3
        figh = plt.figure(figsize=(15, 12))
        gs = mpl.gridspec.GridSpec(2, 2)
        axh1 = figh.add_subplot(gs[0, 0], yscale='log')
        plt.ylabel(f'Proportion of {epitope}-specific clones')
        axh2 = figh.add_subplot(gs[0, 1], yscale='log')
        axh3 = figh.add_subplot(gs[1, 0], yscale='log')
        plt.ylabel(f'Proportion of bulk clones')
        plt.xlabel(f'LD distance from {epitope}-specific clone')
        axh4 = figh.add_subplot(gs[1, 1], yscale='log')
        plt.xlabel(f'TCRDIST distance from {epitope}-specific clone')
        # np.random.seed(110820)
        for tari in np.random.permutation(np.arange(tarmat_tcr.shape[0])):
            """Plots of ECDF using LD and tcrdist"""
            x, y = make_step(ld_thresh, tar_ld_ecdf[tari, :])
            axh1.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)
            
            x, y = make_step(ld_thresh, bulk_ld_ecdf[tari, :])
            axh3.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)

            x, y = make_step(tcr_thresh, tar_tcr_ecdf[tari, :])
            axh2.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)

            x, y = make_step(tcr_thresh, bulk_tcr_ecdf[tari, :])
            axh4.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)
        x, y = make_step(ld_thresh, gmean10(tar_ld_ecdf, axis=0))
        axh1.plot(x, y, color='k', lw=3)
        axh1.set_xlim(ld_lim)

        x, y = make_step(tcr_thresh, gmean10(tar_tcr_ecdf, axis=0))
        axh2.plot(x, y, color='k', lw=3)
        axh2.set_xlim(tcr_lim)

        x, y = make_step(ld_thresh, gmean10(bulk_ld_ecdf, axis=0))
        axh3.plot(x, y, color='k', lw=3)
        axh3.set_xlim(ld_lim)
        axh3.set_ylim((5e-7, 1))

        x, y = make_step(tcr_thresh, gmean10(bulk_tcr_ecdf, axis=0))
        axh4.plot(x, y, color='k', lw=3, alpha=1)
        axh4.set_xlim(tcr_lim)
        axh4.set_ylim((5e-7, 1))
        figh.colorbar(mpl.cm.ScalarMappable(norm=norm_pgen, cmap=mpl.cm.viridis), ax=axh2, label='$P_{gen}$')
        figh.colorbar(mpl.cm.ScalarMappable(norm=norm_pgen, cmap=mpl.cm.viridis), ax=axh4, label='$P_{gen}$')
        pdf.savefig(figh)

        """Plot ratio of epitope-specific to bulk ECDFs for both distances"""
        figh = plt.figure(figsize=(15, 6))
        gs = mpl.gridspec.GridSpec(1, 2)
        axh1 = figh.add_subplot(gs[0, 0], yscale='log')
        plt.ylabel(f'Ratio of proportions')
        plt.xlabel(f'LD distance from {epitope}-specific clone')
        axh2 = figh.add_subplot(gs[0, 1], yscale='log')
        plt.xlabel(f'TCRDIST distance from {epitope}-specific clone')

        alpha = 0.1
        for tari in np.random.permutation(np.arange(tarmat_tcr.shape[0])):
            x, y = make_step(ld_thresh, tar_ld_ecdf[tari, :] / bulk_ld_ecdf[tari, :])
            axh1.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)

            x, y = make_step(tcr_thresh, tar_tcr_ecdf[tari, :] / bulk_tcr_ecdf[tari, :])
            axh2.plot(x, y,
                     color=tr.clone_df['pgen_b_color'].iloc[tari],
                     alpha=alpha)

        x, y = make_step(ld_thresh, gmean10(tar_ld_ecdf / bulk_ld_ecdf, axis=0))
        axh1.plot(x, y, color='k', lw=3)
        axh1.set_xlim(ld_lim)
        axh1.set_ylim((0.1, 5e5))

        x, y = make_step(tcr_thresh, gmean10(tar_tcr_ecdf / bulk_tcr_ecdf, axis=0))
        axh2.plot(x, y, color='k', lw=3, alpha=1)
        axh2.set_xlim(tcr_lim)
        axh2.set_ylim((0.1, 5e5))        
        figh.colorbar(mpl.cm.ScalarMappable(norm=norm_pgen, cmap=mpl.cm.viridis), ax=axh2, label='$P_{gen}$')
        pdf.savefig(figh)

        """Plot proportion with at least one neighbor as a function of distance"""
        figh = plt.figure(figsize=(15, 12))
        gs = mpl.gridspec.GridSpec(2, 2)
        axh1 = figh.add_subplot(gs[0, 0], yscale='linear')
        plt.ylabel(f'Proportion of {epitope}-specific clones\nwith \u22651 neighbor')
        axh2 = figh.add_subplot(gs[0, 1], yscale='linear')
        axh3 = figh.add_subplot(gs[1, 0], yscale='linear')
        plt.ylabel(f'Proportion of bulk clones\nwith \u22651 neighbor')
        plt.xlabel(f'LD distance from {epitope}-specific clone')
        axh4 = figh.add_subplot(gs[1, 1], yscale='linear')
        plt.xlabel(f'TCRDIST distance from {epitope}-specific clone')
        
        x, y = make_step(ld_thresh, np.mean(tar_ld_singl, axis=0))
        axh1.plot(x, y, color='k', lw=2)
        axh1.set_xlim(ld_lim)

        x, y = make_step(tcr_thresh, np.mean(tar_tcr_singl, axis=0))
        axh2.plot(x, y, color='k', lw=2)
        axh2.set_xlim(tcr_lim)

        x, y = make_step(ld_thresh, np.mean(bulk_ld_singl, axis=0))
        axh3.plot(x, y, color='k', lw=2)
        axh3.set_xlim(ld_lim)

        x, y = make_step(tcr_thresh, np.mean(bulk_tcr_singl, axis=0))
        axh4.plot(x, y, color='k', lw=2)
        axh4.set_xlim(tcr_lim)
        for a in [axh1, axh2, axh3, axh4]:
            a.set_ylim((0, 1))

        pdf.savefig(figh)