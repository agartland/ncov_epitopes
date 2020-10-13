import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

from fg_shared import *
import pwseqdist
import fishersapi

sys.path.append(opj(_git, 'utils'))
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import get_kmers

sys.path.append(opj(_git))
import HLAPredCache

data_folder = opj(_fg_data, 'ncov_epitopes', 'data', 'immuneCODE_R002')
results_folder = opj(_fg_data, 'ncov_epitopes', 'data', 'mira_hla_2020-SEP-24')

seasonal = ['229E', 'OC43', 'HKU1', 'NL63']
alphabet = 'ARNDCQEGHILKMFPSTWYV'

peptide = 'HTTDPSFLGRY'

md = pd.read_csv(opj(data_folder, 'subject-metadata.csv'), encoding='cp1252')
md = md.loc[md['Cohort'] != 'COVID-19-Exposed']
md = md.dropna(subset=['HLA-A'])

hlaI_cols = ['HLA-A', 'HLA-A.1', 'HLA-B', 'HLA-B.1', 'HLA-C', 'HLA-C.1']
uHLA2 = sorted(set([h[:4] for h in md[hlaI_cols].values.ravel() if not pd.isnull(h)]))
uHLA4 = sorted(set([h[:7].replace(':', '') for h in md[hlaI_cols].values.ravel() if not pd.isnull(h)]))
# os.makedirs(results_folder)
with open(opj(results_folder, 'mira.hla'), 'w') as fh:
    for h in uHLA4:
        if not h in ['C*0409', 'C*0482']:
            fh.write(f'{h}\n')

md = md.assign(hlas2=md.apply(lambda r: '|'.join(r[hlaI_cols].dropna().str.slice(0, 4).unique()), axis=1),
               hlas4=md.apply(lambda r: '|'.join(r[hlaI_cols].dropna().str.slice(0, 7).str.replace(':','').unique()), axis=1),
               cohort=md.Cohort.map({'Healthy (No known exposure)':'Healthy',
                                     'COVID-19-Convalescent':'COVID-19',
                                     'COVID-19-Acute':'COVID-19',
                                     'COVID-19-B-Non-Acute':'COVID-19'}))
md = md.assign(**{h:md['hlas2'].str.contains(h, regex=False) for h in uHLA2})
md = md.assign(**{h:md['hlas4'].str.contains(h, regex=False) for h in uHLA4})
md = md[['Subject', 'Cohort', 'Cell Type', 'cohort', 'Experiment'] + uHLA2 + uHLA4]

ic_pep_fn = opj(data_folder, 'peptide-detail.csv')
icpep = pd.read_csv(ic_pep_fn)
icpep = icpep.rename({'Amino Acids':'peptides'}, axis=1)[['Experiment', 'peptides']].drop_duplicates()
icpep = icpep.assign(count=True)
icpep = icpep.set_index(['Experiment', 'peptides'])['count'].unstack('peptides').fillna(False)

upeps = icpep.columns

df = pd.merge(md, icpep, on='Experiment', how='left')

res = fishersapi.fishers_frame(df, col_pairs=[pair for pair in itertools.product(upeps, uHLA2)])
res = res.loc[res['yval'] & res['xval']].sort_values(['xcol', 'pvalue'])
res = res.assign(cohort='overall',
                 peptides=res['xcal'])

tmp = [res]
for cohort, gby in df.groupby('cohort'):
    tmppeps = [p for p in upeps if gby[p].any()]
    tmphlas = [p for p in uHLA2 if gby[p].any()]
    tmp_res = fishersapi.fishers_frame(gby, col_pairs=[pair for pair in itertools.product(tmppeps, tmphlas)])
    tmp_res = tmp_res.loc[tmp_res['yval'] & tmp_res['xval']].sort_values(['xcol', 'pvalue'])
    tmp_res = tmp_res.assign(cohort=cohort,
                             peptides=tmp_res['xcal'])
    tmp.append(tmp_res)
res = pd.concat(tmp)
res.to_csv(opj(results_folder, f'hla_mira_fishers.csv'), index=False)

hpc = HLAPredCache.hlaPredCache(opj(results_folder, 'mira'), newFile=True)

kmers = pd.read_csv(opj(_fg_data, 'ncov_epitopes', 'data', 'hcov_2020-JUL-30', 'immuneCODE_xreact_nohla.csv'))

sig = res.loc[(res['cohort'] == 'overall') & (res['pvalue'] < 0.01)]

def _parse_mers(peptides):
    peps = peptides.split(',')
    mers = []
    for p in peps:
        for k in [8, 9, 10, 11]:
            mers.extend(get_kmers(p, k))
    return mers

def _ann_mers(xvec, yvec, mers, mismatches=None):
    mers = np.asarray(mers)
    xvec = np.asarray(xvec)
    yvec = np.asarray(yvec)
    L = np.array([len(m) for m in mers])
    i = (xvec < 6) | (yvec < 6)
    for x, y, L, m in zip(xvec[i], yvec[i], L[i], mers[i]):
        plt.annotate(text=f'{m}-{L}', xy=(x, y),
                     textcoords='offset points',
                     xytext=(-5, 5),
                     ha='right',
                     va='bottom',
                     size='x-small')
        if not mismatches is None:
            mismatches = np.asarray(mismatches)
            for x, y, mm in zip(xvec[i], yvec[i], mismatches[i]):
                plt.annotate(text=f'{int(mm)}mm', xy=(x, y),
                             textcoords='offset points',
                             xytext=(0, 0),
                             ha='center',
                             va='center',
                             size='xx-small')

colors = {k:f'C{c}' for k,c in zip(['SARS-CoV-2'] + seasonal, range(5))}
with PngPdfPages(opj(results_folder, 'mira_hla_scatters.pdf'), create_pngs=False) as pdf:
    for peptides, gby in sig.groupby('xcol'):
        # gby = sig.loc[sig['xcol'] == peptides]
        mers = _parse_mers(peptides)
        hlas = gby['ycol'].unique().tolist()
        hlas = list(np.random.permutation([h for h in uHLA4 if h[:4] in hlas and not h in ['C*0409', 'C*0482']]))
        print(peptides, len(hlas))
        if len(hlas) % 2 == 1:
            hlas.append(hlas[0])
        for xh, yh in zip(hlas[::2], hlas[1::2]):
            figh = plt.figure(figsize=(15, 10))
            xvec = np.array([hpc[(xh, m)] for m in mers])
            yvec = np.array([hpc[(yh, m)] for m in mers])
            plt.scatter(xvec,
                        yvec,
                        s=100,
                        color=colors['SARS-CoV-2'],
                        label='SARS-CoV-2',
                        alpha=0.5)
            _ann_mers(xvec, yvec, mers)

            for species, sgby in kmers.loc[kmers['Amino Acids'] == peptides].groupby('match_species'):
                xvec = np.array([hpc[(xh, m)] for m in sgby['match']])
                yvec = np.array([hpc[(yh, m)] for m in sgby['match']])
                plt.scatter(xvec,
                        yvec,
                        s=100,
                        color=colors[species],
                        label=species,
                        alpha=0.5)
                _ann_mers(xvec, yvec, sgby['match'].tolist(), mismatches=sgby['mismatches'].tolist())
            plt.xlabel(f'{xh} binding (log-IC50)')
            plt.ylabel(f'{yh} binding (log-IC50)')
            plt.ylim((0, 11))
            plt.xlim((0, 11))
            plt.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.title(f"{peptides}: {gby['ycol'].unique().tolist()}")
            pdf.savefig(figh)
            plt.close(figh)

