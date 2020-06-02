import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import sys
from os.path import join as opj

from fg_shared import *

sys.path.append(opj(_git, 'utils'))


sys.path.append(opj(_git))
import seqdistance

# sys.path.append(opj(_git, 'artemis'))
"""Selection of 230 peptides from SARS-CoV-2 with predicted binding with A*02, A*24, A*11, B*07, A*01 or A*03.
Predictions are from the 683 predicted by Grifoni, Sidney, ... Sette (Cell Host and Microbe, 2020).
We selected the top ranked binders with weighting towards A*02, A*24, SARS epitope homologues and
surface/structural proteins which may be more likely to be in a vaccine (I don't know what genes are in the Moderna mRNA-1273 vaccine)
"""

source_lu = {'YP_009724397.2 nucleocapsid phosphoprotein':'N',
             'YP_009724389.1 orf1ab polyprotein (pp1ab)':'ORF1ab',
             'YP_009724390.1 surface glycoprotein':'S',
             'YP_009724391.1 ORF3a protein':'ORF3a',
             'YP_009724394.1 ORF6 protein':'ORF6',
             'YP_009724396.1 ORF8 protein':'ORF8',
             'YP_009724392.1 envelope protein':'E',
             'YP_009724393.1 membrane glycoprotein':'M',
             'YP_009724395.1 ORF7a protein':'ORF7a',
             'YP_009725255.1 ORF10 protein':'ORF10'}

prot_lu = {'S':'YP_009724390.1 surface glycoprotein',
           'M':'YP_009724393.1 membrane glycoprotein',
           'N':'YP_009724397.2 nucleocapsid phosphoprotein',
           'Orf 1ab':'YP_009724389.1 orf1ab polyprotein (pp1ab)',
           'ORF1ab':'YP_009724389.1 orf1ab polyprotein (pp1ab)'}

preferred_alleles = {'HLA-A*02:01':80,
                     'HLA-A*24:02':60,
                     'HLA-A*11:01':60,
                     'HLA-B*07:02':40,
                     'HLA-A*01:01':40,
                     'HLA-A*03:01':40}

base_folder = opj(_fg_data, 'ncov_epitopes', 'data')


sette = pd.read_excel(opj(base_folder, 'SARS-CoV-2_HLA-I_prediction.xlsx')) 
cols = ['Allele', 'Start', 'Len', 'Sequence', 'Score', 'Rank', 'Source sequence']

def _name_pep(r):
    template = '{first}{last}{length}-{protein}-ST{start}-{loci}{twodigit}-FH'
    d = dict(first=r['Sequence'][0],
             last=r['Sequence'][-1],
             length=r['Length'],
             protein=r['Protein'],
             start=r['Start'],
             loci=r['Allele'][4],
             twodigit=r['Allele'][6:8])
    return template.format(**d)

final = []
for allele, gby in sette[cols].groupby('Allele'):
    if allele in preferred_alleles:
        tmp = gby.assign(Protein=gby['Source sequence'].map(source_lu),
                         Length=gby['Len'])

        """Stratfiy picking half from S protein"""
        n_s = int(preferred_alleles[allele] / 4)
        n_other = preferred_alleles[allele] - n_s
        s_tmp = tmp.loc[tmp['Protein'] == 'ORF1ab'].sort_values(by='Rank').iloc[:n_s]
        nos_tmp = tmp.loc[tmp['Protein'] != 'ORF1ab'].sort_values(by='Rank').iloc[:n_other]
        tmp = pd.concat((s_tmp, nos_tmp), axis=0)
        tmp = tmp.assign(Name=tmp.apply(_name_pep, axis=1))
        final.append(tmp)

final = pd.concat(final, axis=0, ignore_index=True)

sars = pd.read_csv(opj(base_folder, 'sette_SARS_dominance_table.tsv'), sep='\t', skiprows=1)

keep = sars['HLA Restriction a'].str.contains('02:') | sars['HLA Restriction a'].str.contains('24:')
sars = sars.loc[keep].drop('Sequence', axis=1).rename({'Sequence.1':'Sequence'}, axis=1)
sars = sars.assign(Allele='HLA-A*02:01',
                   Start=sars['Mapped Start–End'].map(lambda s: s.split('-')[0].split('–')[0]),
                   Length=sars['Sequence'].map(len),
                   Score=1,
                   Rank=0)
sars.loc[sars['Sequence'] == 'NFKDQVILL', 'Allele'] = 'HLA-A*24:02'
sars = sars.assign(**{'Source sequence':sars['Protein'].map(prot_lu)})
sars = sars.assign(Protein=sars['Source sequence'].map(source_lu))
sars = sars.assign(Name=sars.apply(_name_pep, axis=1))

dup = final['Name'].isin(sars['Name'].values)
final = final.loc[~dup]

out_cols = ['Name', 'Sequence', 'Length']
keep_cols = out_cols + ['Allele', 'Protein', 'Start', 'Score', 'Rank', 'Source sequence']

final = pd.concat((sars[keep_cols], final[keep_cols]), axis=0, ignore_index=True)
final[keep_cols].to_csv(opj(base_folder, 'fh_ncov_peptides_verbose_2020-MAR-19.csv'), index=False)
final[out_cols].to_csv(opj(base_folder, 'fh_ncov_peptides_2020-MAR-19.csv'), index=False)
final.groupby(['Allele', 'Protein'])['Sequence'].count().to_csv(opj(base_folder, 'fh_ncov_peptides_summary_2020-MAR-19.csv'))

"""
sette = pd.read_excel(opj(base_folder, 'sette_supp_s6.xlsx'), skiprows=1) 
sette = sette.rename({'HLA class I allele restriction':'Allele'}, axis=1)
sette = sette.drop(628)

hla_cols = ['Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14']

cols = ['Protein ID', 'Protein name', 'Peptide start', 'Peptide end', 'Length', 'Peptide', 'Rank']

tmp = [sette[cols + ['Allele']]]
for c in hla_cols:
    ss = sette[cols + [c]].dropna(subset=[c]).rename({c:'Allele'}, axis=1)
    tmp.append(ss)
epdf = pd.concat(tmp, axis=0, ignore_index=True)
epdf = epdf.assign(Rank=epdf['Rank'].map(lambda k: {'n.a.':np.nan}.get(k, k)))

epdf = pd.merge(epdf,
                epdf.groupby('Peptide')['Allele'].count().reset_index().rename({'Allele':'num_alleles'}, axis=1),
                on='Peptide', how='left')

prefdf = epdf.loc[epdf['Allele'].isin(preferred_alleles)]
prefdf.set_index(['Protein name', 'Allele'])"""