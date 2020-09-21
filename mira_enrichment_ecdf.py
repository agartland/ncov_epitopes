import pandas as pd
import numpy as np
from os.path import join as opj
import os
import sys
from functools import partial
from glob import glob
import feather

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.distributions.empirical_distribution import ECDF

from fg_shared import *

import tcrdist as td
from tcrdist.repertoire import TCRrep
from tcrdist.pgen import OlgaModel

sys.path.append(opj(_git, 'utils'))
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from ecdf_helpers import *

ecdf_fn = 'mira_epitope_131_104_HLMSFPQSA_YHLMSFPQSA_ecdfs.feather'
ecdf_fn = 'mira_epitope_130_106_SSNVANYQK_ecdfs.feather'
ecdf_fn = 'mira_epitope_114_144_LPPAYTNSF_ecdfs.feather'
ecdf_fn = 'mira_epitope_7_4448_SEHDYQIGGYTEKW_YQIGGYTEK_YQIGGYTEKW_ecdfs.feather'
ecdf_fn = 'mira_epitope_60_436_MWSFNPETNI_SFNPETNIL_SMWSFNPET_ecdfs.feather'

files = sorted([os.path.split(ff)[1] for ff in glob(opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/ecdfs/*.feather'))])
for ecdf_fn in files:
    print(ecdf_fn)
    e = feather.read_dataframe(opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/ecdfs', ecdf_fn))
    # e = e.rename({'verus':'versus'}, axis=1)
    # print(e.groupby(['name', 'versus', 'metric', 'fclust_thresh'])['thresholds'].count())

    min_pgen = e.loc[e['pgen'] > 0, 'pgen'].min()
    e.loc[e['pgen'] == 0, 'pgen'] = min_pgen
    """Too slow, do at plot time"""
    # e = e.assign(pgen_color=e['pgen'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)))

    refk = 1e5
    repk = int(ecdf_fn.split('_')[3])
    p = np.logspace(np.log10(1/refk), 0, 500)
    repp = (p*repk + 1) / (repk + 1)
    refp = (p*refk + 1) / (refk + 1)

    """Plotting"""
    def plot_one_ecdf(figh, ecdf, thresholds, epitope_name, colors=None, alpha=0.1):
        axh = figh.add_axes([0.15, 0.15, 0.6, 0.7], yscale='log')
        plt.ylabel(f'Proportion of {epitope_name} clones')
        plt.xlabel(f'Distance from {epitope_name} clone')
        if colors is None:
            colors = ['k']*ecdf.shape[0]

        for tari in np.random.permutation(np.arange(ecdf.shape[0])):
            x, y = make_step(thresholds, ecdf[tari, :])
            axh.plot(x, y,
                     color=colors[tari],
                     alpha=alpha)
        x, y = make_step(thresholds, np.mean(ecdf, axis=0))
        axh.plot(x, y,
                 color='r', #tr.clone_df['pgen_b_color'].iloc[tari],
                 alpha=1)
        return axh

    def plot_mean_ecdf(figh, e, gbcols=['versus', 'metric', 'fclust_thresh']):
        axh = figh.add_axes([0.15, 0.15, 0.6, 0.7], yscale='log')
        plt.ylabel(f'Proportion of clones')
        plt.xlabel(f'Distance from clone')
        for gbvals, gby in e.groupby(gbcols):
            emat = gby.set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
            x, y = make_step(emat.columns, np.mean(emat.values, axis=0))
            axh.plot(x, y, alpha=0.6, label='|'.join([str(s) for s in gbvals]))
            lab = '|'.join([e.iloc[0][s] for s in ['versus', 'metric', 'fclust_thresh'] if not s in gbcols])
            plt.title(f"{e['name'].iloc[0]}: {lab}")
        plt.legend(loc=0)
        return axh

    def plot_one_roc(figh, ess, alpha=0.1):
        axh = figh.add_axes([0.15, 0.15, 0.6, 0.7], yscale='log', xscale='log')
        plt.ylabel(f'Proportion of enriched repertoire near cluster')
        plt.xlabel(f'Proportion of reference near cluster')

        if 'tcrdist' in ess['metric'].iloc[0]:
            ann = [12, 25, 50, 75]
        else:
            ann = [1, 2, 3, 4]

        rep = ess.loc[ess['versus'] == 'rep'].set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
        ref = ess.loc[ess['versus'] == 'ref'].set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')

        colors = ess.drop_duplicates(subset=['label', 'pgen'])['pgen'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)).values
        if alpha > 0:
            for tari in np.random.permutation(np.arange(rep.shape[0])):
                x, y = make_step(ref.values[tari, :], rep.values[tari, :])
                axh.plot(x, y,
                         color=colors[tari],
                         alpha=alpha)
        x, y = make_step(np.mean(ref, axis=0), np.mean(rep, axis=0))
        axh.plot(x, y,
                 color='r', #tr.clone_df['pgen_b_color'].iloc[tari],
                 alpha=1)
        x = np.mean(ref, axis=0).values
        y = np.mean(rep, axis=0).values
        for t in ann:
            ti = np.nonzero(rep.columns == t)[0][0]
            plt.plot(x[ti], y[ti], 'ks')
            plt.annotate(xy=(x[ti], y[ti]),
                         text=t,
                         size='xx-small',
                         ha='right',
                         va='bottom')
        xl = plt.xlim()
        yl = plt.ylim()
        mnmx = [ess['ecdf'].min(), ess['ecdf'].max()]
        plt.plot(mnmx, mnmx, '--', color='gray', lw=2)
        plt.plot(refp, repp, '--', color='gray', lw=2)
        plt.ylim(yl)
        plt.xlim(xl)
        return axh

    def plot_mean_roc(figh, ess, gbcols=['versus', 'metric', 'fclust_thresh'], alpha=0.1):
        axh = figh.add_axes([0.15, 0.15, 0.6, 0.7], yscale='log', xscale='log')
        plt.ylabel(f'Proportion of enriched repertoire near cluster')
        plt.xlabel(f'Proportion of reference near cluster')

        for gbvals, gby in ess.groupby(gbcols):
            if 'tcrdist' in gbvals[0]:
                ann = [12, 25, 50, 75]
            else:
                ann = [1, 2, 3, 4]

            rep = gby.loc[gby['versus'] == 'rep'].set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
            ref = gby.loc[gby['versus'] == 'ref'].set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
            x, y = make_step(np.mean(ref, axis=0), np.mean(rep, axis=0))
            axh.plot(x, y, alpha=0.6, label='|'.join([str(s) for s in gbvals]))
            lab = '|'.join([e.iloc[0][s] for s in ['versus', 'metric', 'fclust_thresh'] if not s in gbcols])
            plt.title(f"{e['name'].iloc[0]}: {lab}")
            
            x = np.mean(ref, axis=0).values
            y = np.mean(rep, axis=0).values
            for t in ann:
                ti = np.nonzero(rep.columns == t)[0][0]
                plt.plot(x[ti], y[ti], 'ks')
                plt.annotate(xy=(x[ti], y[ti]),
                             text=t,
                             size='xx-small',
                             ha='right',
                             va='bottom')
        xl = plt.xlim()
        yl = plt.ylim()
        mnmx = [ess['ecdf'].min(), ess['ecdf'].max()]
        plt.plot(mnmx, mnmx, '--', color='gray', lw=2)
        plt.plot(refp, repp, '--', color='gray', lw=2)
        plt.ylim(yl)
        plt.xlim(xl)
        plt.legend(loc=0)
        return axh

    pdf_fn = opj(_fg_data, 'ncov_tcrs', 'adaptive_bio_r2', 'ecdf_2020-AUG-31',
                 ecdf_fn.replace('mira_epitope_', 'M').replace('feather', 'pdf'))
    with PngPdfPages(pdf_fn) as pdf:
        ind = (e['fclust_thresh'] == 0)

        for metric, egby in e.loc[ind].groupby(['metric']):
            figh = plt.figure(figsize=(11, 8))
            axh = plot_mean_ecdf(figh, egby, gbcols=['versus', 'fclust_thresh'])
            pdf.savefig(figh)

        figh = plt.figure(figsize=(11, 8))
        axh = plot_mean_roc(figh, e.loc[ind], gbcols=['metric', 'fclust_thresh'])
        pdf.savefig(figh)
        plt.close(figh)
        
        # emat = e.loc[(e['versus'] == 'rep') & (e['metric'] == 'edit') & (e['fclust_thresh'] == 1)].set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
        for (vs, metric, fclust_thresh), gby in e.loc[ind].groupby(['versus', 'metric', 'fclust_thresh']):
            emat = gby.set_index(['label', 'thresholds'])['ecdf'].unstack('thresholds')
            colors = gby.drop_duplicates(subset=['label', 'pgen'])['pgen'].map(partial(color_lu, norm_pgen, mpl.cm.viridis.colors)).values

            figh = plt.figure(figsize=(11, 8))
            axh = plot_one_ecdf(figh, emat.values, emat.columns, e['name'].iloc[0], colors=colors)
            plt.title(f"{e['name'].iloc[0]} vs. {vs}: {metric} clustered@{fclust_thresh}")
            pdf.savefig(figh)
            plt.close(figh)

        for (metric, fclust_thresh), gby in e.loc[ind].groupby(['metric', 'fclust_thresh']):
            figh = plt.figure(figsize=(11, 8))
            axh = plot_one_roc(figh, gby)
            plt.title(f"{gby['name'].iloc[0]}: {metric} clustered@{fclust_thresh}")
            pdf.savefig(figh)
            plt.close(figh)