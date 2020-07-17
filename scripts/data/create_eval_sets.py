import csv
import glob
import os
import sys

from collections import defaultdict

from semvecpy.vectors import semvec_utils as sp

# version 7: 72053427 sentences.
CORPUS_FILE = 'data/corpus/medline_triples_v7.txt'

WORDVECS = 'vectors/skipgram/embeddingvectors.bin'
SEMVECS = 'vectors/esd/semanticvectors.bin'

PATH_TO_KB_DUMPS = sys.argv[1]

# Load vectors
svecs = set(sp.readfile(SEMVECS)[0])
wvecs = set(sp.readfile(WORDVECS)[0])

# Set intersection: terms for which vector exists in both spaces
vecs = svecs.intersection(wvecs)

with open(CORPUS_FILE, 'r') as inf:
    lines = [line.split('\t') for line in inf.readlines()]

# Count up term cooccurrence in full corpus
pair_count_dict = defaultdict(int)
for line in lines:
    term_1 = line[0].lower().replace(" ", "_")
    term_2 = line[4].lower().replace(" ", "_")
    pair = (term_1, term_2)
    # Ensure vectors for both terms are available in both vector spaces
    if (term_1 in vecs and term_2 in vecs):
        pair_count_dict[pair] +=1

# Write out corpus cooccurrence counts
cooccur_counts = sorted(pair_count_dict.items(), key=lambda x: x[1], reverse=True)
with open('data/corpus/pair_counts.tsv', 'w+') as outf:
    for pair, i in cooccur_counts:
        outf.write('{0}\t{1}\t{2}\n'.format(pair[0], pair[1], i))

# Produce evaluation sets from knowledge base data
raw_pair_counts = defaultdict(dict)
for f in glob.glob('{0}/*'.format(PATH_TO_KB_DUMPS)):
    pairs = set([tuple(row) for row in csv.reader(open(f, 'r'), delimiter='\t')])
    kb_name = os.path.splitext(os.path.basename(f))[0]
    for pair in pairs:
        # skip weirdly formatted pairs
        if len(pair) == 2:
            # check for presence in vector spaces
            if (pair[0] in vecs and pair[1] in vecs):
                # This count includes reverse order cooccurrence.
                count = pair_count_dict[(pair[0], pair[1])]
                count += pair_count_dict[(pair[1], pair[0])]
                raw_pair_counts[kb_name][pair] = count

outdir = "data/eval_pairs_with_counts"
if not os.path.exists(outdir):
    os.makedirs(outdir)

for kb, terms in raw_pair_counts.items():
    kb_file = "{0}.txt".format(kb)
    outfile = os.path.join(outdir, kb_file)
    with open(outfile, 'w+') as outf:
        pairs = sorted(terms.items(), key=lambda x: x[1], reverse=True)
        for pair, i in pairs: 
            # Skip pairs with same term twice.
            if pair[0] == pair[1]:
                continue
            outf.write('{0}\t{1}\t{2}\n'.format(pair[0], pair[1], i))
