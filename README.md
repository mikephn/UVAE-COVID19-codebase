## UVAE: Integration of Unpaired and Heterogeneous Clinical Flow Cytometry Data
### associated codebase.

This repository contains the code used in the paper to extract data, train, and benchmark models.

Data files including training data, models, and results are accessible on Zenodo: `https://doi.org/10.5281/zenodo.13854783`.

You will also need the UVAE framework from `https://github.com/mikephn/UVAE` placed in the project folder.

- [1 Extracting flow cytometry samples](#1-extracting-flow-cytometry-samples)
- [2 Generating synthetic data](#2-generating-synthetic-data)
- [3 Benchmarking with synthetic data](#3-benchmarking-with-synthetic-data)
- [4 Comparison with cyCombine](#4-comparison-with-cyCombine)
- [5 Training data integration models](#5-training-data-integration-models)
- [6 Training longitudinal regression models](#6-training-longitudinal-regression-models)
- [7 Cross-validation models](#7-cross-validation-models)

## 1 Extracting flow cytometry samples

`extract_flow.R` was used to process .fcs files from a number of experiments. The script allows for subsampling flow cytometry events either from raw files or extracting corrected and annotated files from .wsp workspaces. Rare cell-type sub-populations can be up-sampled and separately stored. Combined dataset is stored in a .hdf5 file. 

`data/lineage.h5` and `data/chemo.h5` contain combined, sub-sampled, annotated patient data from lineage and chemokine panels, respectively, saved in HDF5 format. They can be read with a class `HdfDataset` from `HdfDataset.py`, which provides functions to obtain series of data filtered by common feature sets.

`data/channels-rename.tsv` contains assignment of experimental channels to a common namespace of flow features and markers. This assignment was used to combine analogous features for further processing.

`data/meta.csv` contains assignments of samples to anonymised patient IDs, peak disease severity, processing batch, and timepoints.

## 2 Generating synthetic data

`ToyDataUnsup.py` contains the script to generate synthetic data from a source flow file `data/WB_main 20200504_YA HC WB_main_018.fcs`. Synthetic data is set to contain 3 panels (with disparate features), with 3 batches in each. The assignment of GMM clustering from `scikit-learn` is used to unevenly split the data across batches.

The synthetic dataset is saved as `data/ToyDataWB-3x3.pkl`.

## 3 Benchmarking with synthetic data

`UVAE_test_configs.py` contains a list of model configurations which were tested. These configurations contain settings recognised by the training script to create a particular model configuration.

`UVAE_test.py` is the training script for instantiating a particular model configuration, training it, and saving test results. It trains the ground truth models when one of the first 3 configs is specified.

## 4 Comparison with cyCombine

CLL samples were downloaded from `http://flowrepository.org/id/FR-FCM-Z52G`, Van Gassen data samples were downloaded from `https://flowrepository.org/id/FR-FCM-Z247` and processed with cyCombine functions. The processing scripts and resulting .RDS files are available on Zenodo (`https://doi.org/10.5281/zenodo.13854783`).

`train-uvae-cc-comp-alignment.py` contains the script used for training the UVAE models for EMD/MAD benchmarks on public data.

`cycomb_synth_imp.R` uses cyCombine to impute the missing channels on the synthetic dataset, and `cc_synth_test_scores.py` calculates the imputation MSE, and cLISI/iLISI/EMD/MAD in the data space.

The external datasets (DFCI and van Gassen) are used for additional imputation benchmarking. The process of splitting the datasets and hiding markers is implemented in `prep_imputation_data.R`. cyCombine is run using the `cycomb_synth_imp.R` script. Various UVAE configurations are run with `train-imputation.py`. The metrics are calculated with `imputation_scoring.py`. The full result files are available on Zenodo (`https://doi.org/10.5281/zenodo.13854783`).

## 5 Training data integration models

Two models are trained, one for lineage panels, and one for neutrophils from the chemokine panel. For both models, a number of the largest panels are first integrated. Hyper-parameters are optimised with the search scores saved to `hyper_lisi.csv`. Then the remaining panels are trained to project on top of the initial latent space.

`train-lineage.py` contains the script for training the lineage model. It is first run with the `test` variable set to False, then repeated with it set to True. The resulting model is saved as `model_lineage.uv`. After training, a subset of corrected data is saved to `embs/lineage.pkl`, which is used in subsequent training of the longitudinal regression models.

`train-chemo-neutro.py` contains the script for training the neutrophil chemokine model. The resulting model is saved as `model_chemo.uv`. A subset of corrected data is saved to `embs/chemo.pkl`. For this dataset, Gaussian Mixture Models are used to cluster the data, and are saved in the `gmm` folder.

## 6 Training longitudinal regression models

`train-severity-reg.py` contains a script for training and evaluating individual regression models. The list of `configs` defines each model configuration, a configuration is selected by providing a variable externally or setting the `config_ind` variable. After training the models, setting `config_ind` to None will aggregate all results into a table. By setting `gradientAttribution` to True, gradients of severity will be collected during prediction for patient IDs defined in `pids_to_plot` (by default, PIDs with number of timesteps >= `plot_min_timesteps`, set to 3 are included).

## 7 Cross-validation models

In addition to models trained on the whole dataset, longitudinal regression models are trained by holding out patient samples from the initial UVAE integration. For each fold, two separate UVAE models are trained (one for lineage, one for chemokine), which first integrate the training data, perform clustering of neutrophils, then project the held out data and clustering labels. The UVAE models are trained using scripts in `crossvalidation/train-lineage-split.py` and `crossvalidation/train-chemo-neutro-split.py` and generate separate embeddings for each fold. `crossvalidation/train-severity-reg-held.py` is then used to train the regression models on the resulting data.