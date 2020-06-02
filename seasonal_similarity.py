import matplotlib as mpl
import seaborn as sns

from fg_shared import *
import sys

import skbio
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
from seqtools import *

sys.path.append(opj(_git, 'ncov_epitopes'))
from helpers import *

proj_folder = opj(_fg_data, 'ncov_epitopes')
hcov_fn = opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'all_hcov.fasta')

hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_hcov.csv'))
ds_hcov = pd.read_csv(opj(proj_folder, 'data', 'hcov_2020-MAY-29', 'processed_thinned_hcov.csv'))

k = 9
strains = ds_hcov.strain.unique()
sub_ind = ds_hcov['nsymbol'] == 's'
sim_dict = create_sim_dict(ds_hcov.loc[sub_ind, 'seq'].tolist(), mm_thresh=1, k=k)

strain_seqs = [tuple([st] + ds_hcov.loc[sub_ind & (ds_hcov.strain == st), 'seq'].tolist()) for st in strains]
strain_pw = pw_kmer_strain_similarity(strain_seqs, k=k, sim_func=jaccard_similarity, sim_dict=sim_dict)


plt.figure(1)
plt.clf()
sns.heatmap(strain_pw, vmin=0, vmax=1, square=True)