import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt


from fg_shared import *
import pwseqdist

sys.path.append(opj(_git, 'utils'))
from pngpdf import PngPdfPages

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import get_kmers

sys.path.append(opj(_git))
import HLAPredCache

sys.path.append(opj(_git, 'tcrdist3'))
from tcrdist.repertoire import TCRrep

data_folder = opj(_fg_data, 'ncov_epitopes', 'data', 'm1_hla_2020-SEP-23')

seasonal = ['229E', 'OC43', 'HKU1', 'NL63']

peptide = 'HTTDPSFLGRY'

alphabet = 'ARNDCQEGHILKMFPSTWYV'

hcov_fn = opj(_fg_data, 'ncov_epitopes', 'data', 'hcov_2020-JUL-30', 'selected_hcov_fullkmer_xreact.csv')
hcov = pd.read_csv(hcov_fn)
hcov = hcov.loc[(hcov['species'] == 'SARS-CoV-2') & hcov['match_species'].isin(seasonal)]

sars2_kmers = []
for k in [8, 9, 10, 11]:
    sars2_kmers.extend(get_kmers(peptide, k))

hcov = hcov.loc[hcov['kmer'].isin(sars2_kmers)]

hcov_kmers = hcov['match'].unique().tolist()

def get_all_subs(p):
    out = []
    for i in range(len(p)):
        for aa in alphabet:
            out.append(p[:i] + aa + p[i+1:])
    return out

variants = []
for mer in hcov_kmers + sars2_kmers:
    variants.extend(get_all_subs(mer))

hcov_mers = {species:hcov.loc[hcov['match_species'] == species, 'match'].unique().tolist() for species in seasonal}
hcov_mers['SARS-CoV-2'] = list(set(sars2_kmers))
hcov_mers['variant'] = list(set([v for v in variants if not v in hcov_kmers + sars2_kmers]))

for k in [8, 9, 10, 11]:
    with open(opj(data_folder, f'm1hla.{k}.mers'), 'w') as fh:
        for mer in variants:
            if len(mer) == k:
                fh.write(f'{mer}\n')

hpc = HLAPredCache.hlaPredCache(opj(data_folder, 'm1hla'), newFile=True)

hlas = ['A*0101',
        'A*1101',
        'A*0201',
        'B*0801',
        'C*0602',
        'C*0701',
        'A*0301',
        'A*2402',
        'B*0702',
        'B*1501',
        'B*4001',
        'B*4402',
        'B*5102',
        'C*0304',
        'C*0401',
        'C*0501',
        'C*0702']
tmp = []
for h in hlas:
    for species in hcov_mers.keys():
        for mer in hcov_mers[species]:
            tmp.append({'mer':mer,
                        'hla':h,
                        'ic50':hpc[(h, mer)],
                        'species':species})
df = pd.DataFrame(tmp)

plotdf = df.set_index(['species', 'mer', 'hla'])['ic50'].unstack('hla')
with PngPdfPages(opj(data_folder, 'm1_hla_scatters.pdf'), create_pngs=False) as pdf:
    # for x, y in itertools.product(hlas, hlas):
    x, y = ('A*0101', 'B*1501')
    for y in hlas[1:]:
        figh = plt.figure(figsize=(15, 10))
        for i, gby in plotdf.groupby('species'):
            if i == 'variant':
                plt.scatter(gby[x], gby[y], label=i, alpha=0.1, s=5, color='gray')
            else:
                plt.scatter(gby[x], gby[y], label=i, alpha=1, s=100)
                for ii, r in gby.iterrows():
                    plt.annotate(text=len(ii[1]),
                                 xy=(r[x], r[y]),
                                 ha='center',
                                 va='center',
                                 size='xx-small')
                    if (r[[x, y]] < 6).any():
                        plt.annotate(text=ii[1],
                                     xy=(r[x], r[y]),
                                     ha='right',
                                     va='bottom',
                                     size='xx-small',
                                     textcoords='offset points',
                                     xytext=(-5, 5))
        plt.xlabel(f'{x} binding (log-IC50)')
        plt.ylabel(f'{y} binding (log-IC50)')
        plt.ylim((0, 11))
        plt.xlim((0, 11))
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        pdf.savefig(figh)
        plt.close(figh)
