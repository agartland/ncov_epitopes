import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys

from fg_shared import *

# sys.path.append(opj(_git, 'utils'))
# sys.path.append(opj(_git))
# import HLAPredCache

# sys.path.append(opj(_git, 'tcrdist2'))
# from tcrdist.repertoire import TCRrep

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import get_kmers

proj_folder = opj(_fg_data, 'ncov_epitopes')

ic_pep_fn = opj(proj_folder, 'data', 'immuneCODE_R002', 'peptide-hits.csv')

hom_pep_fn = opj(proj_folder, 'figures', 'hcov_kmer_cross-reactivity_2020-JUN-12.csv')
hom_pep_nohla_fn = opj(proj_folder, 'data', 'hcov_2020-JUL-30', 'selected_hcov_fullkmer_xreact.csv')

icpep = pd.read_csv(ic_pep_fn)

comma = icpep['Amino Acids'].str.contains(',')

wocomma = icpep.loc[~comma]
wocomma = wocomma.assign(peptide=wocomma['Amino Acids'], pepi=0)

"""Expand the comma separated peptides in Amino Acids"""
def _expand(r):
    tmp = [r.to_frame().T.assign(peptide=p, pepi=i) for i,p in enumerate(r['Amino Acids'].split(','))]
    df = pd.concat(tmp, axis=0)
    return df
res = []
for i, r in icpep.loc[comma].iterrows():
    res.append(_expand(r))
wcomma = pd.concat(res, axis=0)
exp_comma = pd.concat((wocomma, wcomma), axis=0, sort=False)

"""Expand the peptides into ALL child kmers"""
def _expand_mers(r, kvec):
    tmp = []
    for k in kvec:
        tmp.extend([r.to_frame().T.assign(kmer=p, kmeri=i, k=k) for i, p in enumerate(get_kmers(r['peptide'], k))])
    df = pd.concat(tmp, axis=0)
    return df

tmp = []
for i, r in exp_comma.iterrows():
    tmp.append(_expand_mers(r, kvec=[8, 9, 10, 11]))
kmers = pd.concat(tmp, axis=0, sort=False)


"""Join with the homology data (including HLA)"""
hom = pd.read_csv(hom_pep_fn)
hom.loc[:, 'mismatches'] = hom.loc[:, 'mismatches'].astype(int).map(str)
hom_cols = ['gene', 'nsymbol', 'kmer',
            'match_species', 'match_nsymbol', 'match', 'mismatches',
            'hla', 'kmer_ic50', 'match_ic50', 'delta_ic50']

out = pd.merge(kmers,
                hom[hom_cols],
                how='left',
                on='kmer')

summ = out[['kmer', 'k', 'ORF', 'match_species', 'mismatches']].fillna('>3').groupby(['k', 'mismatches', 'match_species'])['kmer'].count()
print(summ)
out.to_csv(opj(proj_folder, 'data', 'hcov_2020-JUL-30', 'immuneCODE_xreact_whla.csv'))

"""Join with the homology data (including HLA)"""
tmp = []
for i, r in exp_comma.iterrows():
    tmp.append(_expand_mers(r, kvec=[8, 9, 10, 11, 12, 13, 14, 15]))
kmers = pd.concat(tmp, axis=0, sort=False)

hom = pd.read_csv(hom_pep_nohla_fn)

seasonal = ['229E', 'OC43', 'HKU1', 'NL63']
ind = (hom['species'] == 'SARS-CoV-2') & (hom['mismatches'] <= 5) & (hom['match_species'] != 'SARS-CoV-2') & hom['match_species'].isin(seasonal)
hom = hom.loc[ind]

hom.loc[:, 'mismatches'] = hom.loc[:, 'mismatches'].astype(int).map(str)
hom_cols = ['gene', 'nsymbol', 'kmer',
            'match_species', 'match_nsymbol', 'match', 'mismatches']

out = pd.merge(kmers,
                hom[hom_cols],
                how='left',
                on='kmer')

summ = out[['kmer', 'k', 'ORF', 'match_species', 'mismatches']].fillna('>5').groupby(['k', 'mismatches', 'match_species'])['kmer'].count()
print(summ)
out.to_csv(opj(proj_folder, 'data', 'hcov_2020-JUL-30', 'immuneCODE_xreact_nohla.csv'))
