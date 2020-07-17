# Embedding of Structural Dependencies


Find our trained vectors, trained using the [Semantic Vectors](https://github.com/semanticvectors/semanticvectors) package, on [Zenodo](https://zenodo.org/record/3832324). 

## Setup

Run this script to download Version 7 of the [corpus](https://zenodo.org/record/3459420#.XxDtIZNKiRs) of raw dependency paths released by Percha & Altman. 

```
scripts/setup/get_paths_v7.sh
```

Next, run this script to produce corpus files with which to train vector space models:

```
scripts/setup/create_corpus_files.py
```

This will generate `medline_fullsents_v7.txt`, which contains *just* the full sentences from the corpus (for training, e.g., a skip-gram model), and `medline_triples_v7.txt`, which contains concept-path-concept triples and other metadata (for training ESD). 

## Train vector spaces

