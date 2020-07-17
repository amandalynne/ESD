# Embedding of Structural Dependencies


Find our trained vectors, trained using the [Semantic Vectors](https://github.com/semanticvectors/semanticvectors) package, on [Zenodo](https://zenodo.org/record/3832324). To train the models yourself, follow these steps:

## Setup

Run this script to download Version 7 of the [corpus](https://zenodo.org/record/3459420#.XxDtIZNKiRs) of raw dependency paths released by Percha & Altman. 

```
scripts/setup/get_paths_v7.sh
```
These will get saved to `data/paths/`.

Next, run this script to produce corpus files with which to train vector space models:

```
scripts/setup/create_corpus_files.py
```

This will generate `data/corpus/medline_fullsents_v7.txt`, which contains *just* the full sentences from the corpus (for training, e.g., a skip-gram model), and `data/corpus/medline_triples_v7.txt`, which contains concept-path-concept triples and other metadata (for training ESD). 

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

This will produce .bin files in `vectors/skipgram`.

### Train Embedding of Structural Dependencies:

```
bash scripts/train/train_esd.sh
```

This will produce .bin files in `vectors/esd`.

## Evaluation

For a Python interface to the trained vectors, we use the [Semvecpy](https://github.com/semanticvectors/semvecpy) package. 

### Evaluation data

Scripts for downloading and post-processing knowledge base data are forthcoming!
For now, this script assumes you have a local copy of the knowledge base dumps 
from which to generate evaluation data sets for each knowledge base, consisting 
of pairs of terms for which there exists a vector in both trained vector spaces. 

```
python scripts/data/create_eval_sets.py PATH_TO_KB_DUMPS
``` 
