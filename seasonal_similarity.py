import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from fg_shared import *
import sys
import pwseqdist as pwsd

import skbio
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
# from seqtools import *
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import *

proj_folder = opj(_fg_data, 'ncov_epitopes')
hcov_fn = opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'all_hcov.fasta')

hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_hcov.csv'))
ds_hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_thinned_hcov.csv'))
sel_hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'selected_hcov.csv'))

k = 9
symbols = ['s', 'm' , 'e', 'n']
gene_lu = {'s':'Spike', 'm':'Membrane', 'e':'Envelope', 'n':'Nucleocapsid'}
strains = sel_hcov.strain.unique()

"""Use exact matching to look at kmer similarities"""
with PngPdfPages(opj(proj_folder, 'figures', 'sel_kmer_similarity.pdf')) as pdf:
    for nsymbol in symbols:
        sub_ind = sel_hcov['nsymbol'] == nsymbol

        strain_seqs = [tuple(sel_hcov.loc[sub_ind & (sel_hcov.strain == st), 'seq'].tolist()) for st in strains]
        strain_pw = pw_kmer_strain_similarity(strain_seqs, k=k, sim_func=jaccard_similarity)
        strain_pw = pd.DataFrame(strain_pw, index=strains, columns=strains)

        figh = plt.figure(1, figsize=(15, 8))
        sns.heatmap(strain_pw, vmin=0, vmax=1, square=True)
        plt.title(gene_lu[nsymbol])
        pdf.savefig(figh)
        plt.close(figh)

# sub_ind = ds_hcov['nsymbol'] == 's'
# seqs = ds_hcov.loc[sub_ind, 'seq'].tolist()[:10]
# kmers = list(set([mer for s in seqs for mer in get_kmers(s, k=k)]))
# pw_kmers = pwsd.numba_tools.nb_pairwise_sq(kmers, nb_metric=pwsd.numba_tools.nb_hamming_distance)

def cross_kmer_search(hcov, k):
    gene_lu = {'s':'spike', 'm':'membrane', 'e':'envelope', 'n':'nucleocapsid'}
    res = []
    for sp, gby in hcov.groupby('species'):
        for gene in gby['gene'].unique():
            sub_ind = gby['gene'] == gene
            nsymbol =  gby.loc[sub_ind, 'nsymbol'].iloc[0]
            seqs = gby.loc[sub_ind, 'seq'].tolist()
            kmers = list(set([mer for s in seqs for mer in get_kmers(s, k=k)]))
            if len(kmers) == 0:
                """Solved a problem for very short genes <15aa"""
                continue
            mat = seqs2mat(kmers, alphabet=alphabet)

            for sp_m, gby_m in hcov.groupby('species'):
                #if not sp == sp_m:
                kmers_m = []
                for gene_m in gby_m.gene.unique():
                    sub_ind = gby_m['gene'] == gene_m
                    seqs = gby_m.loc[sub_ind, 'seq'].tolist()
                    kmers_tmp = list(set([mer for s in seqs for mer in get_kmers(s, k=k)]))
                    kmers_m.append(pd.DataFrame({'kmers':kmers_tmp,
                                                 'gene':gene_m,
                                                 'nsymbol':gby_m.loc[sub_ind, 'nsymbol'].iloc[0]}))
                kmers_df = pd.concat(kmers_m, axis=0)
                kmers_m = kmers_df['kmers'].tolist()
                genes_m = kmers_df['gene'].tolist()
                nsymbol_m = kmers_df['nsymbol'].tolist()
                mat_m = seqs2mat(kmers_m, alphabet=alphabet)
                mres = closest_match(mat, mat_m)

                tmp = {'species':sp,
                       'gene':gene,
                       'nsymbol':nsymbol,
                       'kmer':kmers,
                       'match_species':sp_m,
                       'match_gene':[genes_m[mres[i, 0]] if mres[i, 0]>=0 else '' for i in range(mres.shape[0])],
                       'match_nsymbol':[nsymbol_m[mres[i, 0]] if mres[i, 0]>=0 else '' for i in range(mres.shape[0])],
                       'match':[kmers_m[mres[i, 0]] if mres[i, 0]>=0 else '' for i in range(mres.shape[0])],
                       'mismatches':mres[:, 1]}
                tmp = pd.DataFrame(tmp)
                res.append(tmp)
    res = pd.concat(res, axis=0)
    return res

res9 = cross_kmer_search(sel_hcov, k=9)
res15 = cross_kmer_search(sel_hcov, k=15)

def _summarize(res, ind, agg_func, k, mm, fillna, vmax):
    gby_cols = ['species', 'gene', 'match_species']
    if ind is None:
        tmp = res
    else:
        tmp = res.loc[ind]
    if agg_func is np.mean:
        prop_str = 'Proportion'
    elif agg_func is np.sum:
        prop_str = 'Number'
    """Aggregating across kmers within species, gene, and match_species"""
    summ = tmp.groupby(gby_cols)['mismatches'].agg(lambda v: agg_func(v < mm)).unstack('match_species')
    summ = summ.drop('SARS-CoV-2', axis=1, errors='ignore')
    if fillna:
        summ = summ.fillna(1)
    
    figh = plt.figure(figsize=(7, 8))
    sns.heatmap(summ.loc['SARS-CoV-2'], vmin=0, vmax=vmax, square=True)
    plt.ylabel('SARS-CoV-2 gene')
    plt.xlabel('Comparator species (all 9mers)')
    if mm == 1:
        plt.title('%s of SARS-CoV-2 %dmers\nwith an exact match' % (prop_str, k))
    else:
        plt.title('%s of SARS-CoV-2 %dmers\nwith a match sharing >=%d AAs' % (prop_str, k, k - mm + 1))
    return figh

seasonal = ['229E', 'OC43', 'HKU1', 'NL63']
with PngPdfPages(opj(proj_folder, 'figures', 'sel_kmer_mismatch.pdf')) as pdf:
    for res,k in zip([res9, res15], [9, 15]):
        for mm in [1, 2, 3]:
            figh = _summarize(res,
                              ind=None,
                              agg_func=np.mean,
                              k=k,
                              mm=mm,
                              fillna=True,
                              vmax=None)
            pdf.savefig(figh)
            plt.close(figh)

            figh = _summarize(res,
                              ind=None,
                              agg_func=np.sum,
                              k=k,
                              mm=mm,
                              fillna=False,
                              vmax=None)
            pdf.savefig(figh)
            plt.close(figh)

            figh = _summarize(res,
                              ind=res['match_species'].isin(seasonal),
                              agg_func=np.mean,
                              k=k,
                              mm=mm,
                              fillna=True,
                              vmax=None)
            pdf.savefig(figh)
            plt.close(figh)

            figh = _summarize(res,
                              ind=res['match_species'].isin(seasonal),
                              agg_func=np.sum,
                              k=k,
                              mm=mm,
                              fillna=False,
                              vmax=None)
            pdf.savefig(figh)
            plt.close(figh)


ind = (res9.species=='SARS-CoV-2') & (res9['mismatches'] <= 2) & (res9['match_species'] != 'SARS-CoV-2') & (res9['match_species'].isin(seasonal))
tmp = res9.loc[ind].sort_values(by=['mismatches', 'kmer']).drop_duplicates(['kmer', 'match'], keep='first')
tmp.to_csv(opj(proj_folder, 'figures', 'hcov_9mer_cross-reactivity_2020-JUN-03.csv'))

ind = (res15.species=='SARS-CoV-2') & (res15['mismatches'] <= 2) & (res15['match_species'] != 'SARS-CoV-2') & (res15['match_species'].isin(seasonal))
tmp = res15.loc[ind].sort_values(by=['mismatches', 'kmer']).drop_duplicates(['kmer', 'match'], keep='first')
tmp.to_csv(opj(proj_folder, 'figures', 'hcov_15mer_cross-reactivity_2020-JUN-03.csv'))