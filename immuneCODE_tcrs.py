import pandas as pd
import numpy as np
from os.path import join as opj
import itertools
import sys

from fg_shared import *

sys.path.append(opj(_git, 'utils'))
sys.path.append(opj(_git))
import HLAPredCache

sys.path.append(opj(_git, 'tcrdist2'))
from tcrdist.repertoire import TCRrep

sys.path.append(opj(_git, 'ncov_epitopes'))

proj_folder = opj(_fg_data, 'ncov_epitopes')


tr = TCRrep(cell_df=pd.DataFrame(),
            organism='human', 
            chains=['beta'])
tr.rebuild(dest_tar_name='legacy.tar.gz')
# tr.rebuild(dest_tar_name=opj(proj_folder, 'data', 'immuneCODE_R001', 'legacy.tar.gz'))