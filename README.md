# Embedding of Structural Dependencies


Find our trained vectors, trained using the [Semantic Vectors](https://github.com/semanticvectors/semanticvectors) package, on [Zenodo](https://zenodo.org/record/3832324). To train the models yourself, follow these steps:

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

You will need to download and install the [Semantic Vectors](https://github.com/semanticvectors/semanticvectors) package for this. 
(You may need to also install Maven!)

```
git clone https://github.com/semanticvectors/semanticvectors.git
cd semanticvectors
mvn install -P endUserRelease
```

The following scripts train models with the parameters used for our paper. See the Semantic Vectors documentation if you'd like to change these.

### Train a skip-gram model:

```
bash scripts/train/train_skipgram.sh
```

### Train Embedding of Structural Dependencies:

```
bash scripts/train/train_esd.sh
```
