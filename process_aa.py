import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys
import skbio

from fg_shared import *

sys.path.append(opj(_git, 'utils'))
sys.path.append(opj(_git))
import HLAPredCache

sys.path.append(opj(_git, 'ncov_epitopes'))


proj_folder = opj(_fg_data, 'ncov_epitopes')
base_folder = opj(proj_folder, 'data')


aaseqs = []
for s in skbio.read(opj(base_folder,'seq_2020-MAR-22', 'ProteinFastaResults.fasta'), format='fasta'):
    protein = s.metadata['description'].split('|')[-1].split(':')[-1]
    aaseqs.append(dict(protein=protein, seq=str(s)))
seqs = pd.DataFrame(aaseqs)
seqs = seqs.assign(protein=seqs['protein'].map(lambda s: s.replace(' protein', '').replace(' polyprotein', '').upper()))
ren = {'ENVELOPE': 'E',
       'MEMBRANE GLYCOPROTEIN':'M',
       'NUCLEOCAPSID PHOSPHOPROTEIN':'N',
       'SURFACE GLYCOPROTEIN':'S'}

seqs = seqs.assign(protein=seqs['protein'].map(lambda s: ren.get(s,s)),
                   length=seqs['seq'].map(len))
keep_protein = ['E', 'M', 'N', 'ORF1AB', 'S', 'ORF10', 'ORF3A', 'ORF6', 'ORF7A', 'ORF8']
keep_lengths = {'E': 75,
                 'M': 222,
                 'N': 419,
                 'ORF10': 38,
                 'ORF1AB': 7096,
                 'ORF3A': 275,
                 'ORF6': 61,
                 'ORF7A': 121,
                 'ORF8': 121,
                 'S': 1273}
seqs = seqs.loc[seqs['protein'].isin(keep_protein)]
seqs = seqs.loc[seqs.apply(lambda r: r['length'] == keep_lengths[r['protein']], axis=1)]


"""Find the columns with mutations"""
'''INCOMPLETE
for prot, gby in seqs.groupby('protein'):
    mat = np.asarray([[aa for aa in s] for s in gby['seq']])
    diffcols = [i for i in range(mat.shape[1]) if np.any(mat[0, i] != mat[:, i])]
    if len(diffcols) > 0:
        printcols = []
        for i in diffcols:
            #printcols.extend(range(i-2, i+2))
            printcols.extend([i])
        out = np.unique([''.join(mat[i, printcols]) for i in range(mat.shape[0])])
        print(prot)
        for o in out:
            print(o)
        print()
'''
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
immuneCODE_hlas = ['A*2901', 'B*1501', 'B*5701', 'C*0303', 'C*0701',
                  'C*0401', 'C*0501',
                  'A*7401', 'B*3512',
                  'B*4427', 'C*0702', 'C*0704',
                  'A*3201', 'B*1401', 'C*0802',
                  'A*3301', 'B*1402', 'C*0501',
                  'B*0706', 'B*3802', 'C*0727',
                  'B*2705', 'B*4001', 'C*0304', 'C*0704']
                  
tmp = []
for s in seqs['seq']:
    tmp.extend(HLAPredCache.getMers(s))
tmp = list(set(tmp))

"""List of peptides is assembled from SARS-CoV-2 plus some variants
Adding matches (up to 3 mismatches) from other CoV species"""
xreact = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'selected_hcov_xreact.csv'))

ind = (xreact['species'] == 'SARS-CoV-2') & (xreact['mismatches'] <= 3) & (xreact['match_species'] != 'SARS-CoV-2')
tmpx = xreact.loc[ind].drop_duplicates(['kmer', 'match'], keep='first')
tmp.extend(tmpx['kmer'].tolist())
tmp.extend(tmpx['match'].tolist())

tmp = [pep for pep in set(tmp) if not 'X' in pep and not '*' in pep]

for nmer in [8, 9, 10, 11]:
    with open(opj(base_folder, 'ncov.%d.mers' % nmer), 'w') as fh:
        for p in tmp:
            if len(p) == nmer:
                fh.write('%s\n' % (p))
with open(opj(base_folder, 'ncov.hla'), 'w') as fh:
    for k in hla_freq.keys():
        fh.write('%s\n' % k)
    for k in immuneCODE_hlas:
        fh.write('%s\n' % k)

"""
./iedb_predict.py --method netmhcpan \
                  --pep /fh/fast/gilbert_p/fg_data/ncov_epitopes/data/ncov.9.mers \
                  --hla /fh/fast/gilbert_p/fg_data/ncov_epitopes/data/ncov.hla \
                  --out /fh/fast/gilbert_p/fg_data/ncov_epitopes/data/ncov.9.out \
                  --verbose --cpus 4
"""
"""TODO:
 - Predict binding to the Sette alleles
 - Identify epitope hotspots
 - Find if mutations are in epitope hotspots: more than chance?
 - Make plots of hotspots and mutations for each protein
 - Check for Sette peptide variants
 - Process NT sequences and look at locations of snyonymous and NS mutations relative to epitope hotspots"""