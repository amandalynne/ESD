import glob
import os
import re
import sys

sys.path.append(os.getcwd())

# All the dependency path files 
files = glob.glob('data/paths/part-ii-*.txt')

if files:
    pass
else:
    sys.exit("No data!\
             You probably need to run scripts/setup/get_paths_v7.sh first.")

lines = []
for f in files:
    with open(f, 'r+') as inf:
        lines.extend(inf.readlines())

# Write full sentences to file (to train, e.g., skip-gram)
sent_file_path = "data/corpus/medline_fullsents_v7.txt"
os.makedirs(os.path.dirname(sent_file_path), exist_ok=True)
with open(sent_file_path, 'w+') as outf:
    for line in lines:
        l = line.split('\t')
        sentence = l[-1]
        outf.write(sentence)

# Write concept-path-concept triples to file (to train ESD)
trip_file_path = "data/corpus/medline_triples_v7.txt"
os.makedirs(os.path.dirname(trip_file_path), exist_ok=True)
with open(trip_file_path, 'w+') as outf:
    for line in lines:
        l = line.split('\t')
        pmid = l[0]
        sentence = l[-1]
        # Replace pipe with underscore in dep. path segments
        path = re.sub('\|', '_', l[-2])
        ent_1 = l[6]
        ent_2 = l[7]
        # Also include entity IDs and PubTator types
        ent_1_id = l[8]
        ent_2_id = l[9]
        ent_1_type = l[10]
        ent_2_type = l[11]
        outline = '\t'.join([ent_1, ent_1_id, ent_1_type, path, ent_2, ent_2_id, ent_2_type, pmid, sentence])
        outf.write(outline)
