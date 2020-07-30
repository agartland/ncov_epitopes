import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys

from fg_shared import *

sys.path.append(opj(_git, 'utils'))
sys.path.append(opj(_git))
import HLAPredCache

sys.path.append(opj(_git, 'ncov_epitopes'))


proj_folder = opj(_fg_data, 'ncov_epitopes')

res = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'selected_hcov_xreact.csv'))

hpc = HLAPredCache.hlaPredCache(opj(proj_folder, 'data', 'ncov'), newFile=True)

seasonal = ['229E', 'OC43', 'HKU1', 'NL63']
ind = res['k'].isin([8, 9, 10, 11]) & (res['species'] == 'SARS-CoV-2') & (res['mismatches'] <= 3) & (res['match_species'] != 'SARS-CoV-2') & (res['match_species'].isin(seasonal))
tmp = res.loc[ind].sort_values(by=['mismatches', 'kmer']).drop_duplicates(['kmer', 'match'], keep='first')
"""Twelve alleles with worldwide prevalence over 6% (Grifoni, Cell Host & Microbe, 2020)"""
hla_freq = {'A*0101':0.162,
            'A*0201':0.252,
            'A*0301':0.154,
            'A*1101':0.129,
            'A*2301':0.064,
            'A*2402':0.168,
            'B*0702':0.133,
            'B*0801':0.115,
            'B*3501':0.065,
            'B*4001':0.103,
            'B*4402':0.092,
            'B*4403':0.076}
hlas = list(hla_freq.keys())

def _best_allele(r):
    ic50 = [hpc[(h, r['kmer'])] for h in hlas]
    mini = np.argmin(ic50)
    h = hlas[mini]
    ic50_match = hpc[(h, r['match'])]
    out = dict(hla=h,
               kmer_ic50=ic50[mini],
               match_ic50=ic50_match,
               delta_ic50=ic50_match - ic50[mini])
    return pd.Series(out)

best = tmp.apply(_best_allele, axis=1)
tmp = pd.concat([tmp, best], axis=1)

tmp.to_csv(opj(proj_folder, 'figures', 'hcov_kmer_cross-reactivity_2020-JUN-12.csv'))

ind = (res['k'] == 15) & (res['species'] == 'SARS-CoV-2') & (res['mismatches'] <= 2) & (res['match_species'] != 'SARS-CoV-2') & (res['match_species'].isin(seasonal))
tmp = res.loc[ind].sort_values(by=['mismatches', 'kmer']).drop_duplicates(['kmer', 'match'], keep='first')
tmp.to_csv(opj(proj_folder, 'figures', 'hcov_15mer_cross-reactivity_2020-JUN-03.csv'))


"""
figh = plt.figure()
thresh = np.log(50)
params = dict(kde=False, norm_hist=False)
sns.distplot(tmp.loc[(tmp['mismatches'] == 1) & (tmp['kmer_ic50'] < thresh), 'delta_ic50'], label='1 AA mismatches', **params)
sns.distplot(tmp.loc[(tmp['mismatches'] == 2) & (tmp['kmer_ic50'] < thresh), 'delta_ic50'].dropna(), label='2 AA mismatches', **params)
sns.distplot(tmp.loc[(tmp['mismatches'] == 3) & (tmp['kmer_ic50'] < thresh), 'delta_ic50'].dropna(), label='3 AA mismatches', **params)
plt.xlabel('Change in predicted HLA binding affinity, among IC50 < 500 nM\n(log-fold increase in nM)')
plt.xlim([-5, 10])
plt.legend(loc=0)
plt.ylabel('Number of SARS-CoV-2\nhomologous kmers')
figh.savefig(opj(proj_folder, 'figures', 'delta_ic50_dist.png'))
"""