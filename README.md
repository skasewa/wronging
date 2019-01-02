# Wronging a Right: Generating Better Errors to Improve Grammatical Error Detection

This is a repository for the code used in our paper "Wronging a Right: Generating Better Errors to Improve Grammatical Error Detection", by S. Kasewa, P. Stenetorp, and S. Riedel, EMNLP 2018.

The repository is essentially a fork of the following two repos, subsequently referred to as 'source' repos:
  - https://github.com/marekrei/sequence-labeler
  - https://github.com/google/seq2seq/
  
and as such, all licenses, and terms and conditions from these sources apply. Further, please refer back to these sources for more information about the individual repos, as they are quite popular and well-documented.


### Requriements

This project was tested with:
- Python 3.5.2
- Tensorflow 1.12.0
- Numpy 1.14.0
- Matplotlib 3.0.2


### Installation

Ideally, set up a virtual environment with the above requirements, and then clone this repository with:

    git clone https://github.com/skasewa/wronging.git


### Data

You will need to acquire source data in order to run these experiments. You can request for the original FCE dataset from [here](https://www.ilexir.co.uk/datasets/index.html), and you can request the original authors of the respective papers for the [FCE parallel corpus](https://aclweb.org/anthology/P/P11/P11-1019.pdf) and the [FCE token-labelled dataset](http://aclweb.org/anthology/P/P16/P16-1112.pdf).


### NMT training and corruption

Example scripts given [here](https://github.com/skasewa/wronging/tree/master/seq2seq/scripts) demonstrate how to train the NMT system, and generate from it using each of the three methods: argmax, temperature sampling, and beam search.

For more information about the NMT system, please see the [README.md in the seq2seq folder](https://github.com/skasewa/wronging/tree/master/seq2seq). This repository mainly adds temperature sampling, and a copy-mechanism to the beam search to the seq2seq source repo. 


### Utils

[This file](https://github.com/skasewa/wronging/blob/master/utils/tsvutils.py) implements essential tools such as Levenshtein distance and alignment, in order to generate token-labelled training data from the original and NMT-corrupted sentence pairs. It also contains wrappers for automatically processing the NMT outputs into the `.tsv` format used by the sequence labeler.


### Sequence Labelling

Example configs given [here](https://github.com/skasewa/wronging/tree/master/sequence-labeler/conf) demonstrate how to train this modified sequence labeler. Specifically, we extend the sequence labeler to train in batches the size of the main (unaugmented) dataset.

For more information about running the sequence labeler, please see the [README.md in the sequence-labeler folder](https://github.com/skasewa/wronging/tree/master/sequence-labeler). This has minor modifications from its source repo.


### References

This work is described here:

[**Wronging a Right: Generating Better Errors to Improve Grammatical Error Detection**](http://aclweb.org/anthology/D18-1541)   
Sudhanshu Kasewa and Pontus Stenetorp and Sebastian Riedel   
In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP-2018)*   

For further references, please consult the source repositories.



