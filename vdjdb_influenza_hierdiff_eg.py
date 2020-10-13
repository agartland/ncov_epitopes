""" 
2020-09-25
AN EXAMPLE ILLUSTRATES CONTROLLING EACH STEP MANUALLY
"""

"""imports are listed in the order that they are used, and left in the code below for clarity"""
import os
from os.path import join as opj
import pandas as pd
import json
from tcrdist.mappers import vdjdb_to_tcrdist2, vdjdb_to_tcrdist2_mapping_TRA, vdjdb_to_tcrdist2_mapping_TRB
from tcrdist.repertoire import TCRrep
from tcrdist.rep_diff import hcluster_diff, member_summ
from tcrdist.tree import TCRtree
from tcrsampler.sampler import TCRsampler
from palmotif import compute_pal_motif, svg_logo
from tcrdist.summarize import _select
from hierdiff import plot_hclust_props
from fg_shared import _fg_data

proj_folder = opj(_fg_data, 'ncov_epitopes', 'data', 'vdjdb_examples')
raw = pd.read_csv(os.path.join(proj_folder, 'influenza_2020-SEP.tsv'), sep = '\t')

"""PREPROCESSING INPUT DATA (SEE DOCS PAGE: https://tcrdist3.readthedocs.io/en/latest/inputs.html)"""
pmids = ['PMID:25609818', 'PMID:28423320']

selin = raw.loc[raw['Reference'] == pmids[0]]
selin = selin.assign(cohort=selin['Meta'].map(lambda d: json.loads(d)['subject.cohort']),
                     ptid=selin['Meta'].map(lambda d: json.loads(d)['subject.id']))

from tcrdist.mappers import vdjdb_to_tcrdist2, vdjdb_to_tcrdist2_mapping_TRA, vdjdb_to_tcrdist2_mapping_TRB
selin_a = selin.loc[selin['Gene'] == 'TRA'].rename(vdjdb_to_tcrdist2_mapping_TRA, axis=1)
selin_b = selin.loc[selin['Gene'] == 'TRB'].rename(vdjdb_to_tcrdist2_mapping_TRB, axis=1)

"""COMPUTE TCRDISTANCES (SEE DOCS PAGE: https://tcrdist3.readthedocs.io/en/latest/tcrdistances.html)"""
from tcrdist.repertoire import TCRrep
tr = TCRrep(cell_df=selin_a,
            organism='human',
            chains=['alpha'])

"""COMPUTE TCRDISTANCES (SEE DOCS PAGE:https://tcrdist3.readthedocs.io/en/latest/index.html#hierarchical-neighborhoods)"""
from tcrdist.rep_diff import hcluster_diff
tr.hcluster_df, tr.Z =\
    hcluster_diff(clone_df = tr.clone_df, 
                  pwmat    = tr.pw_alpha,
                  x_cols = ['cohort'], 
                  count_col = 'count')

"""
SEE TCRSAMPLER (https://github.com/kmayerb/tcrsampler/blob/master/docs/tcrsampler.md)
Here we used olga human alpha synthetic sequences for best coverage
"""
from tcrsampler.sampler import TCRsampler
t = TCRsampler()
#t.download_background_file('olga_sampler.zip') # ONLY IF NOT ALREADY DONE
tcrsampler_alpha = TCRsampler(default_background = 'olga_human_alpha_t.sampler.tsv')
tcrsampler_alpha.build_background(max_rows = 1000) 

"""SEE PALMOTIF DOCS (https://github.com/agartland/palmotif)"""
from palmotif import compute_pal_motif, svg_logo
from tcrdist.summarize import _select

"""GENERATE SVG GRAPHIC FOR EACH NODE OF THE TREE"""
pwmat_str = 'pw_alpha'
cdr3_name = 'cdr3_a_aa'
gene_names = ['v_a_gene','j_a_gene']
svgs_alpha = list()
svgs_alpha_raw = list()
for i,r in tr.hcluster_df.iterrows():
    dfnode   = tr.clone_df.iloc[r['neighbors_i'],].copy()
    # <pwnode> Pairwise Matrix for node sequences
    pwnode   = getattr(tr, pwmat_str)[r['neighbors_i'],:][:,r['neighbors_i']].copy()
    if dfnode.shape[0] > 2:
        iloc_idx = pwnode.sum(axis = 0).argmin()
        centroid = dfnode[cdr3_name].to_list()[iloc_idx]
    else:
        centroid = dfnode[cdr3_name].to_list()[0]
    
    print(f"ALPHA-CHAIN CENTROID: {centroid}")
    
    gene_usage_alpha = dfnode.groupby(gene_names).size()
    sampled_rep = tcrsampler_alpha.sample( gene_usage_alpha.reset_index().to_dict('split')['data'], 
                    flatten = True, depth = 10)
    
    sampled_rep  = [x for x in sampled_rep if x is not None]

    motif, stat = compute_pal_motif(
                    seqs = _select(df = tr.clone_df, 
                                   iloc_rows = r['neighbors_i'], 
                                   col = cdr3_name),
                    refs = sampled_rep, 
                    centroid = centroid)
    svgs_alpha.append(svg_logo(motif, return_str= True))

    sampled_rep = sampled_rep.append(centroid)
    motif_raw, _ = compute_pal_motif(
                seqs =_select(df = tr.clone_df, 
                               iloc_rows = r['neighbors_i'], 
                               col = cdr3_name),
                centroid = centroid)
    svgs_alpha_raw.append(svg_logo(motif_raw, return_str= True))  

"""Add Alpha SVG graphics to hcluster_df"""
tr.hcluster_df['svg_alpha'] = svgs_alpha
tr.hcluster_df['svg_alpha_raw'] = svgs_alpha_raw

"""
SUMMARIZE EACH NODE
members_summ summarize the gene usage and other categorical variables within each node
"""
from tcrdist.rep_diff import hcluster_diff, member_summ
res_summary = member_summ(  res_df = tr.hcluster_df,
                            clone_df = tr.clone_df, 
                            addl_cols=['cohort'])
"""hcluster_df_detailed will provide the final set of information used to make an interactive tree graphic"""
tr.hcluster_df_detailed = \
    pd.concat([tr.hcluster_df, res_summary], axis = 1)
"""GENERATE HTML FOR INTERACTIVE GRAPHIC"""
from hierdiff import plot_hclust_props
html = plot_hclust_props(tr.Z,
    title='INFLUENZA EXAMPLE',
    res=tr.hcluster_df_detailed,
    tooltip_cols=['cdr3_a_aa','v_a_gene', 'j_a_gene','svg_alpha', 'svg_alpha_raw'],
    alpha=0.001, colors = ['blue','gray'],
    alpha_col='pvalue')
"""WRITE HTML TO A DISK"""
with open('hierdiff_influenza_example.html', 'w') as fh:
    fh.write(html)