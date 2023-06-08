"""
This script does the following:
    - Load in vectors from trained models
    - Load in pre-processed evaluation data extracted from knowledge bases (KBs)
    - Construct an analogical ranked retrieval task (A is to B as C is to D)
    - For a given evaluation pair (C, D):
        - Select K cue pairs (A, B) of the same relation type (i.e. from the same KB) 
        - Create a list L of terms and their accompanying vectors such that
            they do not co-occur with C in any training corpora
        - Create a query vector (B-A+C) and produce a nearest neighbor ranking
            of the vectors in L 
        - Record the rank of target term D in this list
    - Do this for all (C, D) pairs and for values [1, 10, 25] of cue size K
"""


import csv
import glob
import numpy as np
import os
import random
import sys

import semvecpy.vectors.semvec_utils as sv
import semvecpy.vectors.binary_vectors as bv
import semvecpy.vectors.real_vectors as rv

from collections import defaultdict
from datetime import datetime
from itertools import product
from multiprocess import Pool
from numpy import median


# Load and initialize vector spaces

# SGNS
WORDVECS = '/envme/paullada/medline_intersect_exps/sgns/embeddingvectors.bin'

wvecs=rv.RealVectorStore()
wvecs.init_from_file(WORDVECS)
wvecs.normalize_all()

# Dependency paths (full path)
ESD_SVECS = '/envme/paullada/medline_intersect_exps/esd/semanticvectors.bin'
ESD_EVECS = '/envme/paullada/medline_intersect_exps/esd/elementalvectors.bin'

svecs_esd=bv.BinaryVectorStore()
svecs_esd.init_from_file(ESD_SVECS)
evecs_esd=bv.BinaryVectorStore()
evecs_esd.init_from_file(ESD_EVECS)

# SemRep triples
SR_SVECS = '/envme/paullada/medline_intersect_exps/semrep/semanticvectors.bin' 
SR_EVECS = '/envme/paullada/medline_intersect_exps/semrep/elementalvectors.bin' 

svecs_sr=bv.BinaryVectorStore()
svecs_sr.init_from_file(SR_SVECS)
evecs_sr=bv.BinaryVectorStore()
evecs_sr.init_from_file(SR_EVECS)

# Dependency + SemRep triples (non-segmented)
J_SVECS = '/envme/paullada/JBI_vectors/jointNOSEG/semanticvectors.bin'
J_EVECS = '/envme/paullada/JBI_vectors/jointNOSEG/elementalvectors.bin'

svecs_joint=bv.BinaryVectorStore()
svecs_joint.init_from_file(J_SVECS)
evecs_joint=bv.BinaryVectorStore()
evecs_joint.init_from_file(J_EVECS)

# Dependency triples (segmented) 
ESD_SEG_SVECS = '/envme/paullada/JBI_vectors/dataSEG/semanticvectors.bin'
ESD_SEG_EVECS = '/envme/paullada/JBI_vectors/dataSEG/elementalvectors.bin'

svecs_esd_seg=bv.BinaryVectorStore()
svecs_esd_seg.init_from_file(ESD_SEG_SVECS)
evecs_esd_seg=bv.BinaryVectorStore()
evecs_esd_seg.init_from_file(ESD_SEG_EVECS)

# Dependency + SemRep triples (segmented)
J_SEG_SVECS = '/envme/paullada/JBI_vectors/jointSEG/semanticvectors.bin'
J_SEG_EVECS = '/envme/paullada/JBI_vectors/jointSEG/elementalvectors.bin'

svecs_joint_seg=bv.BinaryVectorStore()
svecs_joint_seg.init_from_file(J_SEG_SVECS)
evecs_joint_seg=bv.BinaryVectorStore()
evecs_joint_seg.init_from_file(J_SEG_EVECS)

# Create shared term space
d_set = set(svecs_esd.terms)
s_set = set(svecs_sr.terms)
w_set = set(wvecs.terms)

# These are constants that will be used elsewhere
SHARED_TERM_SET = w_set.intersection((d_set.intersection(s_set)))
SHARED_TERMS = list(SHARED_TERM_SET)

def create_shared_space(vectors, vec_type='binary', shared_terms=SHARED_TERMS):
    # Binary vectors are default, other option is 'real' to accommodate w2v
    shared_vecs_list = []
    if vec_type == 'binary':
        shared_vecs = bv.BinaryVectorStore()
    else:
        shared_vecs = rv.RealVectorStore() 

    for term in shared_terms:
        if vec_type == 'binary':
            vec = vectors.get_vector(term).bitset
        else:
            wvec = vectors.get_vector(term).copy()
            wvec.normalize()
            vec = wvec.vector
        shared_vecs_list.append(vec)

    shared_vecs.init_from_lists(shared_terms, shared_vecs_list) 
    return shared_vecs

# Only using terms that occur in *all* vector spaces
shared_wvecs = create_shared_space(wvecs, 'real') 
shared_evecs_esd = create_shared_space(evecs_esd) 
shared_svecs_esd = create_shared_space(svecs_esd)
shared_evecs_sr = create_shared_space(evecs_sr)
shared_svecs_sr = create_shared_space(svecs_sr)
shared_evecs_joint = create_shared_space(evecs_joint)
shared_svecs_joint = create_shared_space(svecs_joint)
shared_evecs_esd_seg = create_shared_space(evecs_esd_seg) 
shared_svecs_esd_seg = create_shared_space(svecs_esd_seg)
shared_evecs_joint_seg = create_shared_space(evecs_joint_seg)
shared_svecs_joint_seg = create_shared_space(svecs_joint_seg)

# Global cooccurrence
pc = defaultdict(int)
with open('/envme/paullada/medline_intersect_exps/global_pair_counts.tsv', 'r+') as inf: 
    lines=inf.readlines()                                                       
    for line in lines:                                                          
        l = line.split('\t')                                                        
        if (l[0] in SHARED_TERM_SET) and (l[1] in SHARED_TERM_SET):
            pair = (l[0], l[1])                                                  
            pc[pair] = int(l[-1])

# Cooccurrence lists per term
COOCCUR_LIST_DICT = defaultdict(set) 
for pair, count in pc.items():                                                    
    pair_a = pair[0]
    pair_b = pair[1]
    if pair_a in SHARED_TERM_SET and pair_b in SHARED_TERM_SET:
        COOCCUR_LIST_DICT[pair_a].add(pair_b)
        # Include reverse order
        COOCCUR_LIST_DICT[pair_b].add(pair_a)

def real_knn_for_cue(c_term, cues, vecs, searchvecs, bd=False):
    # Initialize empty vector
    wvec_cue = rv.RealVectorFactory.generate_zero_vector(len(vecs.vectors[0]))
    for cue in cues:
        if bd:
            # Superpose the B terms directly to the D
            wvec_cue.superpose(vecs.get_vector(cue[1]), 1)
        else:
            wvec = vecs.get_vector(cue[1]).copy()
            wvec.superpose(vecs.get_vector(cue[0]),-1)
            wvec.superpose(vecs.get_vector(c_term), 1)
            wvec.normalize()
            wvec_cue.superpose(wvec,1)

    wvec_cue.normalize()
    knn = searchvecs.knn(wvec_cue, len(searchvecs.vectors))
    res = {pair[1]: i+1 for i,pair in enumerate(knn)}

    return res 


def binary_knn_for_cue(c_term, cues, svecs, evecs, searchvecs, bd=False):
    # Initialize empty vector
    svec_cue = bv.BinaryVectorFactory.generate_zero_vector(svecs.vectors[0].dimension)
    for cue in cues:
        if bd:
            # Superpose the B terms directly to the D
            svec_cue.superpose(svecs.get_vector(cue[1]), 1)
        else:
            svec = svecs.get_vector(cue[0]).copy()
            svec.bind(evecs.get_vector(cue[1]))
            svec.bind(svecs.get_vector(c_term))
            svec_cue.superpose(svec, 1)

    svec_cue.normalize()
    # Compute cues from *full* space (otherwise no term found)
    # But then compute knn from *constrained* space (the vecs are all the same)
    knn = searchvecs.knn(svec_cue, len(searchvecs.vectors))
    res = {pair[1]: i+1 for i,pair in enumerate(knn)}

    return res 


def proc_file(f, num_cues, today, reverse):
    """
    f: file path
    num_cues: int, number of random cues
    today: str, today's date for logging output
    reverse: if True, construct analogy with reversed pairs (D:C::B:A)

    for target pairs (C, D), use <num_cues> (A, B) cues for analogy
    """
    pair_counts = [tuple(row) for row in csv.reader(open(f, 'r'), delimiter='\t')]
    candidates = [(p[0], p[1], p[-1]) for p in pair_counts if (p[0] in SHARED_TERM_SET) and (p[1] in SHARED_TERM_SET)]
    if reverse:
        lbd_pairs = [(p[1], p[0]) for p in candidates if (p[-1] == 'LBD')]
        seen_pairs = [(p[1], p[0]) for p in candidates if (p[-1] == 'RR')]
    else:
        lbd_pairs = [(p[0], p[1]) for p in candidates if (p[-1] == 'LBD')]
        seen_pairs = [(p[0], p[1]) for p in candidates if (p[-1] == 'RR')]

    db_name = os.path.splitext(os.path.basename(f))[0]
    print(db_name)

    c_terms = list(set([x[0] for x in lbd_pairs]))
    
    if reverse:
        lbd_outdir = "/envme/paullada/JBI_vectors/eval_{0}_{1}_lbd_DC".format(today, num_cues)
    else:
        lbd_outdir = "/envme/paullada/JBI_vectors/eval_{0}_{1}_lbd_CD".format(today, num_cues)

    if not os.path.exists(lbd_outdir):
        os.makedirs(lbd_outdir)

    cue_file = '{0}_cues.tsv'.format(db_name)

    # Place to store raw (C, D) ranks
    lbd_rank_dicts = defaultdict(dict)

    # For each C term in pairs...
    for c_term in c_terms:
        # all D terms such that (C, D) are a pair
        lbd_target_dterms = [x[1] for x in lbd_pairs if x[0] == c_term]

        # Cues with entirely different entities, always drawn from 'seen' pairs
        cues_1=[x for x in seen_pairs if x[0] != c_term and x[1] not in lbd_target_dterms]
        cues=[x for x in cues_1 if x[1] != c_term and x[0] not in lbd_target_dterms]

        try:
            # Grab num_cues random cues -- different set for each C
            cue_sample = random.sample(cues, num_cues)
        except:
            continue

        # Place to store raw ranks per d term, per method
        lbd_rank_dict = { d_term: {} for d_term in lbd_target_dterms } 

        # Write out cues and targets
        cues_out = ",".join([" ".join(pair) for pair in cue_sample])
        
        lbd_targets_out = ",".join(lbd_target_dterms)
        with open(os.path.join(lbd_outdir, cue_file), 'a+') as outf:
            outf.write('{0}\t{1}\t{2}\n'.format(c_term, lbd_targets_out, cues_out))

        # All terms that co-occur with this C
        co_terms = list(COOCCUR_LIST_DICT[c_term])

        # Terms that don't co-occur with C
        non_co_terms = [t for t in SHARED_TERMS if t not in co_terms]
        

        # LBD search space consists of terms that DO NOT cooccur with C
        lbd_wvecs = create_shared_space(wvecs, 'real', shared_terms=non_co_terms) 
        # Only need to create elemental vectors here.
        lbd_evecs_esd = create_shared_space(evecs_esd, shared_terms=non_co_terms) 
        lbd_evecs_sr = create_shared_space(evecs_sr, shared_terms=non_co_terms)
        lbd_evecs_joint = create_shared_space(evecs_joint, shared_terms=non_co_terms)
        lbd_evecs_esd_seg = create_shared_space(evecs_esd_seg, shared_terms=non_co_terms) 
        lbd_evecs_joint_seg = create_shared_space(evecs_joint_seg, shared_terms=non_co_terms)

        # find KNN using full "shared" vector spaces 
        w2v_res = real_knn_for_cue(c_term, cue_sample, shared_wvecs, lbd_wvecs) 
        w2v_d_res = real_knn_for_cue(c_term, cue_sample, shared_wvecs, lbd_wvecs, bd=True) 

        esd_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_esd, shared_evecs_esd, lbd_evecs_esd)
        esd_d_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_esd, shared_evecs_esd, lbd_evecs_esd, bd=True)

        sr_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_sr, shared_evecs_sr, lbd_evecs_sr)
        sr_d_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_sr, shared_evecs_sr, lbd_evecs_sr, bd=True)

        joint_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_joint, shared_evecs_joint, lbd_evecs_joint)
        joint_d_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_joint, shared_evecs_joint, lbd_evecs_joint, bd=True)

        esd_seg_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_esd_seg, shared_evecs_esd_seg, lbd_evecs_esd_seg)
        esd_seg_d_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_esd_seg, shared_evecs_esd_seg, lbd_evecs_esd_seg, bd=True)

        joint_seg_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_joint_seg, shared_evecs_joint_seg, lbd_evecs_joint_seg)
        joint_seg_d_res = binary_knn_for_cue(c_term, cue_sample, shared_svecs_joint_seg, shared_evecs_joint_seg, lbd_evecs_joint_seg, bd=True)

        # find ranks for all Ds in ONE Knn search.
        # loop over results in full analogy and BD searches
        for d_term in lbd_target_dterms:
            w2v_rank = w2v_res[d_term]
            w2v_bd_rank = w2v_d_res[d_term]

            esd_rank = esd_res[d_term]
            esd_bd_rank = esd_d_res[d_term]

            sr_rank = sr_res[d_term]
            sr_bd_rank = sr_d_res[d_term]

            joint_rank = joint_res[d_term]
            joint_bd_rank = joint_d_res[d_term]

            esd_seg_rank = esd_seg_res[d_term]
            esd_seg_bd_rank = esd_seg_d_res[d_term]

            joint_seg_rank = joint_seg_res[d_term]
            joint_seg_bd_rank = joint_seg_d_res[d_term]

            lbd_rank_dict[d_term]['w2v'] = w2v_rank 
            lbd_rank_dict[d_term]['w2v_BD'] = w2v_bd_rank 
            lbd_rank_dict[d_term]['ESD'] = esd_rank 
            lbd_rank_dict[d_term]['ESD_BD'] = esd_bd_rank 
            lbd_rank_dict[d_term]['SR'] = sr_rank 
            lbd_rank_dict[d_term]['SR_BD'] = sr_bd_rank 
            lbd_rank_dict[d_term]['JOINT'] = joint_rank 
            lbd_rank_dict[d_term]['JOINT_BD'] = joint_bd_rank 
            lbd_rank_dict[d_term]['ESD_SEG'] = esd_seg_rank 
            lbd_rank_dict[d_term]['ESD_SEG_BD'] = esd_seg_bd_rank 
            lbd_rank_dict[d_term]['JOINT_SEG'] = joint_seg_rank 
            lbd_rank_dict[d_term]['JOINT_SEG_BD'] = joint_seg_bd_rank 
            # Length of the search space
            lbd_rank_dict['len_search_space'] = len(non_co_terms)
    
        lbd_rank_dicts[c_term] = lbd_rank_dict

    raw_ranks_output = '{0}-raw-ranks.tsv'.format(db_name)
    with open(os.path.join(lbd_outdir, raw_ranks_output), 'w+') as outf:
        fields = ['C_TERM',
                  'D_TERM',
                  'SGNS',
                  'SGNS_BD',
                  'ESD',
                  'ESD_BD',
                  'SR',
                  'SR_BD',
                  'JOINT',
                  'JOINT_BD',
                  'ESD_SEG',
                  'ESD_SEG_BD',
                  'JOINT_SEG',
                  'JOINT_SEG_BD',
                  'BASELINE']
        outf.write('\t'.join(fields))
        outf.write('\n')

        for c_term, rank_dict in lbd_rank_dicts.items():
            # Baseline is halfway thru the constrained search space.
            baseline = str(rank_dict['len_search_space'] / 2)
            for d_term in rank_dict.keys():
                if d_term == "len_search_space":
                    continue
                w2v_rank = str(rank_dict[d_term]['w2v'])
                w2v_bd_rank = str(rank_dict[d_term]['w2v_BD'])
                esd_rank = str(rank_dict[d_term]['ESD'])
                esd_bd_rank = str(rank_dict[d_term]['ESD_BD']) 
                sr_rank = str(rank_dict[d_term]['SR'])
                sr_bd_rank = str(rank_dict[d_term]['SR_BD']) 
                joint_rank = str(rank_dict[d_term]['JOINT'])
                joint_bd_rank = str(rank_dict[d_term]['JOINT_BD']) 
                esd_seg_rank = str(rank_dict[d_term]['ESD_SEG'])
                esd_seg_bd_rank = str(rank_dict[d_term]['ESD_SEG_BD']) 
                joint_seg_rank = str(rank_dict[d_term]['JOINT_SEG'])
                joint_seg_bd_rank = str(rank_dict[d_term]['JOINT_SEG_BD']) 
                line = '\t'.join(
                    [c_term,
                     d_term,
                     w2v_rank,
                     w2v_bd_rank,
                     esd_rank,
                     esd_bd_rank,
                     sr_rank,
                     sr_bd_rank,
                     joint_rank,
                     joint_bd_rank,
                     esd_seg_rank,
                     esd_seg_bd_rank,
                     joint_seg_rank,
                     joint_seg_bd_rank,
                     baseline])
                outf.write('{0}\n'.format(line))


if __name__ == "__main__":

    files = glob.glob('/envme/paullada/medline_intersect_exps/eval_pairs/*')
    
    # List of len==1 so can use in product()
    today = [datetime.today().strftime('%m%d%y')]
    
    cue_sizes = [1, 10, 25]
    reverse = [True, False]

    # Start parallel workers at the beginning of script
    pool = Pool(19)
    # Execute computation(s) in parallel
    # Iterate over all combos of file & cue size 
    pool.starmap(proc_file, product(files, cue_sizes, today, reverse)) 
    # Turn off parallel workers at end of script
    pool.close()
