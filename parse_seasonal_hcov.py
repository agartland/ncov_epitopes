import pandas as pd
import numpy as np
from fg_shared import *
import sys

import skbio
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
# from seqtools import *

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
raw = pd.DataFrame(rows)

"""The 4 seasonal coronaviruses should be within these
(OC43 is betacoronavirus 1 I think)"""
keep_species = ['Betacoronavirus_1',
                'Human_coronavirus_229E',
                 'Human_coronavirus_HKU1',
                 'Human_coronavirus_NL63',
                 #'Human_enteric_coronavirus_strain_4408',
                 'Middle_East_respiratory_syndrome_coronavirus',
                 'Severe_acute_respiratory_syndrome_related_coronavirus']
hcov = raw.loc[raw['species'].isin(keep_species)]

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
        if 'Wuhan' in r['strain']:
            """Checked that these are all SARS-CoV-2"""
            return 'SARS-CoV-2'
        return 'SARS'
    else:
        return r['species']

"""Drop betacoronaviruses that don't explicitly say OC43"""
hcov = hcov.loc[~((hcov.species == 'Betacoronavirus_1') & ~hcov.strain.str.contains('OC43') & ~(hcov.genbank_access=='KF923898'))]

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
                    'small_envelope_protein':'e',
                    'surface_glycoprotein':'s',
                    'spike_glycoprotein':'s'})

symbol_na = hcov['nsymbol'] == 'na'
hcov.loc[symbol_na, 'nsymbol'] = hcov.loc[symbol_na, 'gene'].map(lambda k: symbol_lu.get(k, k))

ns_ind = hcov.nsymbol.str.contains('ns') & ~hcov.nsymbol.str.contains('nsp')
hcov.loc[ns_ind, 'nsymbol'] = hcov.loc[ns_ind, 'nsymbol'].str.replace('ns', 'nsp')

hcov.loc[:, 'nsymbol'] = hcov.loc[:, 'nsymbol'].str.replace('_protein', '')

"""For SARS-CoV-2 keep only the strains that meet strict criteria of
having exactly 28 genes in the sequenced genome (n=3228)"""
tmp = hcov.groupby(['species', 'strain', 'subtype', 'country'])['gene'].count().loc['SARS-CoV-2']
keep_sars2 = tmp.loc[tmp == 28].reset_index()['strain'].tolist()
# hcov = hcov.loc[(hcov['species'] != 'SARS-CoV-2') | (hcov['strain'].isin(keep_sars2))]

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

def _strain_name(r):
    if r['strain'].startswith(r['species']):
        return r['strain']
    else:
        return r['species'] + '/' + r['strain']

hcov = hcov.assign(strain=hcov.apply(_strain_name, axis=1))


"""After creating nsymbol I think everything looks good except for classification
of the polyproteins, orfs and non-structural proteins. I'm not sure which is which
or which contains which. It may be that some of these are longer than others too,
which means renaming wouldn't really solve any problems

NOTE
I later found issues with the large number of SARS and SARS-CoV-2 sequences including
miscategorized strains. Also the reference for SARS-CoV-2 was somehow weeded out.
So I've commented out all the filter steps above, leaving the nsymbol renames intact.
Will now try to use all seasonal HCoVs and these select SARS/MERS/SARS2 ref strains:

SARS (Tor2, Urbani includes many double sequences...)
SARS-CoV-2 (Wuhan-Hu-1)
MERS (KNIH/002_05_2015 and HCoV_EMC)"""

# print(hcov[['gene', 'symbol', 'nsymbol']].drop_duplicates().sort_values(by='nsymbol'))

# print(hcov.groupby(['species', 'symbol'])['gene'].count())

tmp = pd.merge(hcov.groupby(['species', 'strain', 'subtype', 'country'])['gene'].count(),
         hcov.groupby(['species', 'strain', 'subtype', 'country'])['L'].sum(),
         left_index=True,
         right_index=True)

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
# hcov = hcov.loc[hcov.apply(lambda r: r['strain_ngene'] in ngene_crit[r['species']], axis=1)]

"""Need to find the refs for the seasonal picks: not as import though since
I will just use as a bag of kmers. Gene category is important though
CHECK THAT"""
ref_strains = {'SARS':['SARS/Tor2'],
               'MERS':['MERS_CoV/KOR/KNIH/002_05_2015',
                       'MERS/HCoV_EMC'],
               'SARS-CoV-2':['SARS-CoV-2/Wuhan_Hu_1'],
               '229E':['229E/human/USA/932_72/1993', # these seem to be the longest 19713 aa
                       '229E/human/USA/933_40/1993',
                       '229E/human/USA/933_50/1993'],
                'OC43':['OC43/HCoV_OC43/Seattle/USA/SC0682/2019', # these were recent, long and consistent 21714 aa
                        'OC43/HCoV_OC43/Seattle/USA/SC0776/2019',
                        'OC43/HCoV_OC43/Seattle/USA/SC0810/2019',
                        'OC43/HCoV_OC43/Seattle/USA/SC0839/2019',
                        'OC43/HCoV_OC43/Seattle/USA/SC0841/2019'
                        'OC43/3184A/2012'], #this is a reference i believe
                'NL63':['NL63/DEN/2008/16',
                        'NL63/human/USA/901_24/1990'],
                'HKU1':['HKU1/human/USA/HKU1_12/2010',
                        'HKU1/human/USA/HKU1_13/2010']}

ref_genbank = {'SARS':['AY274119'],
               'MERS':['JX869059',
                       'KT029139'],
               'SARS-CoV-2':['NC_045512'],
               '229E':['KF514432', # these seem to be the longest 19713 aa
                       'KF514433',
                       'KF514430'],
                'OC43':['MN306036', # these were recent, long and consistent 21714 aa
                        'MN310478',
                        'MN306041',
                        'MN306042',
                        #'MN306043',
                        'KF923898'],
                'NL63':['JQ765566',
                        'KF530111'],
                'HKU1':['KF686346',
                        'KF686343']}

tmp = []
for sp, gby in hcov.groupby('species'):
    gby = gby.loc[gby.genbank_access.isin(ref_genbank[sp])]
    tmp.append(gby)
sel_hcov = pd.concat(tmp, axis=0)

nysmbol_lu = {'small_envelope_e':'e',
              'spike_glycoprotein':'s',
              'nonsptructural_2a':'nsp2a',
              'nonsptructural_5a':'nsp5a'}
sel_hcov = sel_hcov.assign(nsymbol=sel_hcov['nsymbol'].map(lambda s: nysmbol_lu.get(s, s)))
# print(sel_hcov.groupby(['species', 'nsymbol', 'gene', 'L'])['symbol'].count())
for s in seasonal:
    print(s)
    print(hcov.loc[hcov['species'] == s][['species', 'strain', 'strain_ngene', 'strain_L']].drop_duplicates())


"""Downsample to no more than 10 strains per species for comparative analysis"""
nsamples = 10
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
sel_hcov.to_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'selected_hcov.csv'))