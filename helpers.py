import pwseqdist
from functools import partial
import operator
import pandas as pd
import numpy as np
import numba as nb

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def jaccard_similarity(lista, listb, **kwargs):
    if len(lista) == 0 and len(listb) == 0:
        return np.nan
    seta = set(lista)
    setb = set(listb)
    inters = seta.intersection(setb)
    union = seta.union(setb)
    res = len(inters)/len(union)
    return res

def fuzzy_overlap(lista, listb, sim_dict={}):
    i = []
    inters = 0
    for a in seta:
        for b in setb:
            if a == b:
                inters += 1
            else:
                sim = sim_dict.get((a, b), sim_dict.get((b, a), 0))
                if sim == 1:
                    inters += 1

    return inters / (len(lista) + len(listb))/2


def kmer_seq_similarity(s1, s2, k=9):
    sim = jaccard_similarity(get_kmers(s1, k),
                             get_kmers(s2, k))
    return sim

def pw_kmer_similarity(kmers, k=9):
    func = partial(kmer_similarity, k=k)
    pw_sim = pwseqdist.apply_pairwise_rect(kmers, kmers, metric=func, ncpus=1)
    return pw_sim

def kmer_strain_similarity(seqs1, seqs2, k=9, sim_func=jaccard_similarity, sim_dict={}):
    kmers1 = [mer for s in seqs1 for mer in get_kmers(s, k=k)]
    kmers2 = [mer for s in seqs2 for mer in get_kmers(s, k=k)]
    return jaccard_similarity(kmers1, kmers2, sim_dict=sim_dict)

def pw_kmer_strain_similarity(strain_seqs1, k=9, sim_func=jaccard_similarity, sim_dict={}):
    func = partial(kmer_strain_similarity, k=k, sim_func=sim_func, sim_dict=sim_dict)
    pw_sim = pwseqdist.apply_pairwise_rect(strain_seqs1, strain_seqs1, metric=func, ncpus=1)
    return pw_sim

def create_sim_dict(seqs, mm_thresh=1, k=9):
    def _func(mer1, mer2):
        dist = np.sum([i for i in map(operator.__ne__, mer1, mer2)])
        return int(dist <= mm_thresh)

    kmers = list(set([mer for s in seqs for mer in get_kmers(s, k=k)]))
    print('Computing pairwise distances for %d kmers.' % len(kmers))

    sim_dict = {}
    for mer1 in kmers:
        for mer2 in kmers:
            if mer1 < mer2:
                dist = np.sum([i for i in map(operator.__ne__, mer1, mer2)])
                if dist <= mm_thresh:
                    sim = 1
                else:
                    sim = 0
                """If they are the same we don't need to know that they are similar
                and we only need to store one ordering of the pair"""
                sim_dict[(mer1, mer2)] = sim
    return sim_dict


"""Need a table of all kmers, by species, with their closest match
in each of the other species

Build this for each species and each other species at a time."""
alphabet = 'ARNDCQEGHILKMFPSTWYVBZX'
def seqs2mat(seqs, alphabet=alphabet):
    """Convert a collection of AA sequences into a
    numpy matrix of integers for fast comparison.

    Requires all seqs to have the same length."""
    L1 = len(seqs[0])
    mat = np.zeros((len(seqs), L1), dtype=np.int8)
    for si, s in enumerate(seqs):
        assert L1 == len(s), "All sequences must have the same length: L1 = %d, but L%d = %d" % (L1, si, len(s))
        for aai, aa in enumerate(s):
            mat[si, aai] = alphabet.index(aa)
    return mat

def vec2seq(vec, alphabet=alphabet):
    """Convert a numpy array of integers back into a AA sequence.
    (opposite of seq2vec())"""
    return ''.join([alphabet[aai] for aai in vec])

def mat2seqs(mat, alphabet=alphabet):
    """Convert a matrix of integers into AA sequences."""
    return [''.join([alphabet[aai] for aai in mat[i,:]]) for i in range(mat.shape[0])]

@nb.jit(nb.int32[:,:](nb.int8[:, :], nb.int8[:, :]), nopython=True, parallel=True)
def closest_match(kmers, matches):
    """contains index of the closest match
    and the number of mismatches"""
    out = np.zeros((kmers.shape[0], 2), dtype=np.int32)
    for k_i in nb.prange(kmers.shape[0]):
        min_mm = kmers.shape[1]
        min_i = -1
        for m_i in range(matches.shape[0]):
            mm = 0
            for pos_i in range(kmers.shape[1]):
                if kmers[k_i, pos_i] != matches[m_i, pos_i]:
                    mm += 1
            if mm < min_mm:
                min_mm = mm
                min_i = m_i
        out[k_i, 0] = min_i
        out[k_i, 1] = min_mm
        """
        if min_mm <= 1:
            print(kmers[k_i, :], matches[min_i, :], min_mm)
            print(vec2seq(kmers[k_i, :]), vec2seq(matches[min_i, :]))
        """
    return out



