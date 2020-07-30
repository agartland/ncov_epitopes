import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from fg_shared import *
import sys
# import pwseqdist as pwsd

import skbio
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
# from seqtools import *
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import *

proj_folder = opj(_fg_data, 'ncov_epitopes')
hcov_fn = opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'all_hcov.fasta')

# hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_hcov.csv'))
# ds_hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_thinned_hcov.csv'))
sel_hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'selected_hcov.csv'))

# sub_ind = ds_hcov['nsymbol'] == 's'
# seqs = ds_hcov.loc[sub_ind, 'seq'].tolist()[:10]
# kmers = list(set([mer for s in seqs for mer in get_kmers(s, k=k)]))
# pw_kmers = pwsd.numba_tools.nb_pairwise_sq(kmers, nb_metric=pwsd.numba_tools.nb_hamming_distance)

def cross_kmer_search(hcov, k):
    gene_lu = {'s':'spike', 'm':'membrane', 'e':'envelope', 'n':'nucleocapsid'}
    res = []
    for sp, gby in hcov.groupby('species'):
        """Run a search for each kmer in each gene of each species"""
        print('\t%s' % sp)
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
                       'k':k,
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

res = []
for k in range(8, 16):
    print('Searching for homologous %dmers...' % k)
    tmp = cross_kmer_search(sel_hcov, k=k)
    res.append(tmp)

res = pd.concat(res, axis=0)
res.index = range(res.shape[0])

res.to_csv(opj(proj_folder, 'data', 'hcov_2020-JUL-30', 'selected_hcov_fullkmer_xreact.csv'))
