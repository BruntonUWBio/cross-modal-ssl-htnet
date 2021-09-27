# cross-modal-ssl-htnet


## Overview

Self-supervised neural decoding using cross-modal deep clustering.
We show that sharing information across multiple data streams
yields high-quality neural decoders, even when no data labels are
present in the training data.

We follow  the approach from [Alwassel et. al. 2020](https://arxiv.org/abs/1911.12667),
using this [code repository](https://github.com/yukimasano/self-label) from [Asano et. al. 2019](https://arxiv.org/abs/1911.05371) to implement deep clustering.
For the decoder models, we use [HTNet](https://github.com/BruntonUWBio/HTNet_generalized_decoding).


## Citing our paper

If you use our code, please cite our *bioRxiv* [preprint](https://www.biorxiv.org/content/10.1101/2021.09.10.459775v1).

```
Peterson, S. M., Rao, R. P. N., & Brunton, B. W. (2021).
Learning neural decoders without labels using multiple data streams.
bioRxiv. https://www.biorxiv.org/content/10.1101/2021.09.10.459775v1
```


## Replicating our findings

In our paper, we tested this cross-modal, self-supervised approach on
4 datasets. All of these datasets are publicly-available:

ECoG move/rest: https://doi.org/10.6084/m9.figshare.16599782

EEG move/rest: http://bnci-horizon-2020.eu/database/data-sets

ECoG finger flexion: https://searchworks.stanford.edu/view/zk881ps0522

EEG balance perturbations: https://openneuro.org/datasets/ds003739


To replicate our findings,

**1) Convert datasets to xarray files**

Download the data and run the 3 convert_....py scripts (the ECoG
move/rest data just needs to be placed in a similar set of directories as
the other 3 datasets)


**2) Create model hyperparameters**

Run *create_model_params.ipynb* to create the model hyperparameters. The current
values were used in our study.


**3) Train and validate supervised/self-supervised models**

Update the pathnames in all of the bash scripts in the /scripts directory. Then, 
run *run_all_procs.sh* to train and validate model performance for all 4 datasets.
In addition to cross-modal deep clustering, supervised and unimodal deep clustering
models are also trained and validated.


**4) Plot results**

Use *plot_model_performance.ipynb* to replicate Figs. 2-4 from our preprint. Use
*plot_fingerflex_clusters.ipynb* to replicate Fig. S1.


## Funding

This work was supported by funding from the National Science Foundation (1630178 and EEC-1028725), the Washington Research Foundation, and the Weill Neurohub.