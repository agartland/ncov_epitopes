import pandas as pd
import numpy as np
from os.path import join as opj
import sys
# from glob import glob
import os

import feather
import scipy.cluster.hierarchy as sch
import scipy.spatial

# from fg_shared import _fg_data

import pwseqdist as pwsd

import tcrdist as td
from tcrdist.repertoire import TCRrep
from tcrdist.pgen import OlgaModel

import argparse

"""EXAMPLE:
python mira_enrichment_compute_ecdf.py --rep ~/fg_data/ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/pw_computed/mira_epitope_60_436_MWSFNPETNI_SFNPETNIL_SMWSFNPET.tcrdist3.csv --ref ~/fg_data/tcrdist/datasets/human_T_beta_bitanova_unique_clones_sampled_1220K.csv --ncpus 3 --subsample 100
"""

def compute_ecdf(data, counts=None, thresholds=None):
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
    # n_ecdf = (np.sum((data[:, None] <= thresholds[None, :]) * counts[:, None], axis=0) >= n).astype(int)
    return ecdf

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

def run_one(ref_fn, rep_fn, ss=-1, ncpus=1):
    ref_df = pd.read_csv(ref_fn)
    ref_df.columns = [{'v_b_name':'v_b_gene',
                       'j_b_name':'j_b_gene',
                       'cdr3_b_aa':'cdr3_b_aa'}.get(c, c) for c in ref_df.columns]
    ref_df.loc[:, 'count'] = 1
    if ss == -1:
        ref_tr = TCRrep(cell_df=ref_df, 
                        organism='human', 
                        chains=['beta'],
                        compute_distances=False)
    else:    
        ref_tr = TCRrep(cell_df=ref_df.sample(n=ss, replace=False), 
                        organism='human', 
                        chains=['beta'],
                        compute_distances=False)


    """Compute pgen of each MIRA TCR"""
    olga_beta  = OlgaModel(chain_folder="human_T_beta", recomb_type="VDJ")
    ref_tr.clone_df['pgen_cdr3_b_aa'] = olga_beta.compute_aa_cdr3_pgens(tr.clone_df.cdr3_b_aa)

    out = []
    print(rep_fn)
    for metric in ['tcrdist', 'tcrdist-cdr3', 'edit']:
        if 'tcr' in metric:
            metric_thresholds = np.arange(1, 101)
            fcluster_thresholds = [0, 25, 50]
        else:
            metric_thresholds = np.arange(1, 9)
            fcluster_thresholds = [0, 1, 2]

        epitope_name = os.path.split(rep_fn)[1].split('.')[0]
        epitope_name = epitope_name.replace('mira_epitope_', 'M')

        # rep_fn = opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/pw_computed', rep_fn)
        print(f'\t{metric}')
        rep_df = pd.read_csv(rep_fn).assign(count=1)

        tr = TCRrep(cell_df=rep_df[['v_b_gene','j_b_gene','cdr3_b_aa','epitope','experiment','subject', 'count']], 
                    organism='human', 
                    chains=['beta'], 
                    db_file='alphabeta_gammadelta_db.tsv', 
                    compute_distances=False)

        tr.clone_df['pgen_cdr3_b_aa'] = olga_beta.compute_aa_cdr3_pgens(tr.clone_df.cdr3_b_aa)
        
        """with open(rep_fn, 'rb') as fh:
            tr = dill.load(fh)"""

        """Compute repertoire PW distances and create flat clusters"""
        rep_pwmat = _pwrect(tr, clone_df1=tr.clone_df,
                                metric=metric,
                                ncpus=ncpus)

        ref_pwmat = _pwrect(tr, clone_df1=tr.clone_df,
                                clone_df2=ref_tr.clone_df,
                                metric=metric,
                                ncpus=ncpus)

        for fclust_thresh in fcluster_thresholds:
            if fclust_thresh > 0:
                rep_pwvec = scipy.spatial.distance.squareform(rep_pwmat, force='tovector')
                Z = sch.linkage(rep_pwvec, method='complete')
                labels = sch.fcluster(Z, t=fclust_thresh, criterion='distance')
            else:
                labels = np.arange(1, rep_pwmat.shape[0] + 1)

            """Compute ECDF for each cluster within the repertoire"""
            # rep_ecdf = np.zeros((int(np.max(labels)), len(metric_thresholds)))
            for lab in range(1, np.max(labels) + 1):
                lab_ind = labels == lab
                rep_ecdf = compute_ecdf(np.mean(rep_pwmat[lab_ind, :], axis=0), thresholds=metric_thresholds)
                tmp_df = pd.DataFrame({'ecdf':rep_ecdf, 'thresholds':metric_thresholds})
                tmp_df = tmp_df.assign(metric=metric,
                                       fclust_thresh=fclust_thresh,
                                       label=lab,
                                       name=epitope_name,
                                       versus='rep',
                                       pgen=np.median(tr.clone_df['pgen_cdr3_b_aa'].values[lab_ind]))
                out.append(tmp_df)

            """Compute distances to the reference for each cluster and compute ECDF vs reference"""
            # ref_ecdf = np.zeros((int(np.max(labels)), len(metric_thresholds)))
            for lab in range(1, np.max(labels) + 1):
                lab_ind = labels == lab
                ref_ecdf = compute_ecdf(np.mean(ref_pwmat[lab_ind, :], axis=0), thresholds=metric_thresholds)
                tmp_df = pd.DataFrame({'ecdf':ref_ecdf, 'thresholds':metric_thresholds})
                tmp_df = tmp_df.assign(metric=metric,
                                       fclust_thresh=fclust_thresh,
                                       label=lab,
                                       name=epitope_name,
                                       versus='ref',
                                       pgen=np.median(tr.clone_df['pgen_cdr3_b_aa'].values[lab_ind]))
                out.append(tmp_df)
    out = pd.concat(out, axis=0)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ECDFs for a repertoire versus a reference.')
    parser.add_argument('--rep', type=str,
                        help='path to a CSV file containing the seqs for making a TCRRep')
    parser.add_argument('--ref', type=str,
                        help='path to a CSV file containing the reference clones')
    parser.add_argument('--ncpus', type=int,
                        help='number of CPUs to use for distance computation')
    parser.add_argument('--subsample', type=int, default=-1,
                        help='number of clones to use from the reference (reduce to 100 for testing, or -1 for all)')
    args = parser.parse_args()

    # out_folder = opj(_fg_data, 'ncov_tcrs', 'adaptive_bio_r2', 'ecdf_2020-AUG-31')

    """Load the reference sequences"""
    # ref_fn = opj(_fg_data, 'tcrdist/datasets', 'human_T_beta_bitanova_unique_clones_sampled_1220K.csv')
    # olga_ref_fn = opj(_fg_data, 'tcrdist/datasets', 'human_T_beta_sim1000K.csv')

    # rep_filenames = glob(opj(_fg_data, 'ncov_tcrs/adaptive_bio_r2/tcrs_by_mira_epitope/pw_computed/*.dill'))

    """rep_filenames = ['mira_epitope_61_428_GRLQSLQTY_LITGRLQSL_RLQSLQTYV.tcrdist3.csv.dill',
                     'mira_epitope_131_104_HLMSFPQSA_YHLMSFPQSA.tcrdist3.csv.dill',
                     'mira_epitope_60_436_MWSFNPETNI_SFNPETNIL_SMWSFNPET.tcrdist3.csv.dill']"""

    out = run_one(ref_fn=args.ref, rep_fn=args.rep, ss=args.subsample, ncpus=args.ncpus)
    print('Writing results to:', f'{args.rep.replace(".tcrdist3.csv", "")}_ecdfs.feather')
    feather.write_dataframe(out, f'{args.rep.replace(".tcrdist3.csv", "")}_ecdfs.feather')