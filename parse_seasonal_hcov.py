import pandas as pd
import numpy as np
from fg_shared import *
import sys

import skbio
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
from seqtools import *

proj_folder = opj(_fg_data, 'ncov_epitopes')
hcov_fn = opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'all_hcov.fasta')

metadata_cols = ['symbol', 'gene', 'genbank_access',
                 'protein_access', 'uniprot_access', 'strain', 'segment',
                 'date', 'host', 'country',
                 'subtype', 'species']
rows = []
fasta = skbio.io.read(hcov_fn, format='fasta')
for i, seq in enumerate(fasta):
    md = seq.metadata['id'].split('|')
    tmp = {k:v for k,v in zip(metadata_cols, md)}
    tmp.update({'seq':str(seq),
                'L':len(seq)})
    rows.append(tmp)
    
fasta.close()
hcov = pd.DataFrame(rows)

"""The 4 seasonal coronaviruses should be within these
(OC43 is betacoronavirus 1 I think)"""
keep_species = ['Betacoronavirus_1',
                'Human_coronavirus_229E',
                 'Human_coronavirus_HKU1',
                 'Human_coronavirus_NL63',
                 #'Human_enteric_coronavirus_strain_4408',
                 'Middle_East_respiratory_syndrome_coronavirus',
                 'Severe_acute_respiratory_syndrome_related_coronavirus']
hcov = hcov.loc[hcov['species'].isin(keep_species)]

sars2_strains = ['SARS_CoV_2',
                 '2019_nCoV',
                 'hCoV_19',
                 'SAR_CoV_2',
                 'SARS_CoV2',
                 'SARS_Cov2']

def _determine_sars(r):
    if r['species'] == 'Severe_acute_respiratory_syndrome_related_coronavirus':
        for s in sars2_strains:
            if s in r['strain']:
                return 'SARS-CoV-2'
        return 'SARS'
    else:
        return r['species']

"""Drop betacoronaviruses that don't explicitly say OC43"""
hcov = hcov.loc[~((hcov.species == 'Betacoronavirus_1') & ~hcov.strain.str.contains('OC43'))]

species_lu = {'Betacoronavirus_1':'OC43',
              'Human_coronavirus_229E':'229E',
              'Human_coronavirus_HKU1':'HKU1',
              'Human_coronavirus_NL63':'NL63',
              'Human_enteric_coronavirus_strain_4408':'HEC_4408',
              'Middle_East_respiratory_syndrome_coronavirus':'MERS',
              'SARS':'SARS',
              'SARS-CoV-2':'SARS-CoV-2'}

hcov = hcov.assign(species=hcov.apply(_determine_sars, axis=1).map(species_lu),
                   gene=hcov['gene'].str.lower().str.replace('putative_', ''),
                   symbol=hcov['symbol'].str.lower(),
                   nsymbol=hcov['symbol'].str.lower())

symbol_lu = hcov.loc[hcov['symbol']!='na'].set_index('gene')['symbol'].to_dict()
symbol_lu.update({'nonstructural_protein_3':'nsp3',
                    'replicase_1a':'pp1a',
                    'replicase_polyprotein':'pp1ab',
                    'nonstructural_polyprotein_pp1a':'pp1a',
                    'nonstructural_polyprotein_pp1ab':'pp1ab',
                    'envelope':'e',
                    'membrane_glycoprotein_m':'m',
                    'nucleocapsid':'n',
                    'small_envelope_protein':'e'})

symbol_na = hcov['nsymbol'] == 'na'
hcov.loc[symbol_na, 'nsymbol'] = hcov.loc[symbol_na, 'gene'].map(lambda k: symbol_lu.get(k, k))

ns_ind = hcov.nsymbol.str.contains('ns') & ~hcov.nsymbol.str.contains('nsp')
hcov.loc[ns_ind, 'nsymbol'] = hcov.loc[ns_ind, 'nsymbol'].str.replace('ns', 'nsp')

hcov.loc[:, 'nsymbol'] = hcov.loc[:, 'nsymbol'].str.replace('_protein', '')

"""For SARS-CoV-2 keep only the strains that meet strict criteria of
having exactly 28 genes in the sequenced genome (n=3228)"""
tmp = hcov.groupby(['species', 'strain', 'subtype', 'country'])['gene'].count().loc['SARS-CoV-2']
keep_sars2 = tmp.loc[tmp == 28].reset_index()['strain'].tolist()
hcov = hcov.loc[(hcov['species'] != 'SARS-CoV-2') | (hcov['strain'].isin(keep_sars2))]

uniprot_lu = {'U3M6F4':'nsp5a',
              'Q6W361':'nsp2',
              'B7TYG7':'nsp2a',
              'B7TYH0':'nsp1',
              'B7TYH1':'nsp4'}

def _polyproteins(r):
    if '1ab' in r['nsymbol'] or '1ab' in r['gene']:
        return 'pp1ab'
    elif '1a' in r['nsymbol'] or '1a' in r['gene']:
        return 'pp1a'
    elif r['uniprot_access'] in uniprot_lu:
        return uniprot_lu[r['uniprot_access']]
    else:
        return r['nsymbol']
hcov = hcov.assign(nsymbol=hcov.apply(_polyproteins, axis=1))

"""After creating nsymbol I think everything looks good except for classification
of the polyproteins, orfs and non-structural proteins. I'm not sure which is which
or which contains which. It may be that some of these are longer than others too,
which means renaming wouldn't really solve any problems"""
print(hcov[['gene', 'symbol', 'nsymbol']].drop_duplicates().sort_values(by='nsymbol'))

print(hcov.groupby(['species', 'symbol'])['gene'].count())

tmp = pd.merge(hcov.groupby(['species', 'strain', 'subtype', 'country'])['gene'].count(),
         hcov.groupby(['species', 'strain', 'subtype', 'country'])['L'].sum(),
         left_index=True,
         right_index=True)
for s in hcov.species.unique():
    print(s)
    print(tmp.loc[s])

tmp = tmp.rename({'gene':'strain_ngene',
                  'L':'strain_L'}, axis=1)

hcov = pd.merge(hcov,
                tmp.reset_index()[['species', 'strain', 'strain_ngene', 'strain_L']],
                on=['species', 'strain'], how='left')

"""Since table of total lengths by strain shows outliers, use these criteria
for number of genes for a strain"""
ngene_crit = {'229E':[23],
              'HKU1':[24],
              'NL63':[23],
              'OC43':[21, 23],
              'MERS':[27, 28],
              'SARS':[27, 30],
              'SARS-CoV-2':[28]}
hcov = hcov.loc[hcov.apply(lambda r: r['strain_ngene'] in ngene_crit[r['species']], axis=1)]

def _strain_name(r):
    if r['strain'].startswith(r['species']):
        return r['strain']
    else:
        return r['species'] + '/' + r['strain']

hcov = hcov.assign(strain=hcov.apply(_strain_name, axis=1))

"""Downsample to no more than 10 strains per species for comparative analysis"""
nsamples = 2
tmp = []
np.random.seed(110820)
for sp, gby in hcov.groupby('species'):
    strains = gby.strain.unique()
    if len(strains) > nsamples:
        strains = strains[np.random.permutation(len(strains))[:nsamples]]
        gby = gby.loc[gby.strain.isin(strains)]
    tmp.append(gby)
ds_hcov = pd.concat(tmp, axis=0)

print(ds_hcov.groupby(['species', 'strain', 'strain_ngene', 'strain_L'])['nsymbol'].count())

hcov.to_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_hcov.csv'))
ds_hcov.to_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_thinned_hcov.csv'))