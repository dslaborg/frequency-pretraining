# Experiment Descriptions

* exp001 (explore number of subjects)
    * exp001a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (each
      with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001b: pre-training + downstream training with frozen FE, a Gridsearch over the following hyperparameter
      combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001c: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001d: no pre-training + downstream training with frozen FE, a Gridsearch over the following hyperparameter
      combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
* exp002 (explore number of epochs and subjects)
    * exp002a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (
      each with 3 different seeds): n_epochs in [-1,50,130,340,900], n_subjects in [1,2,3,4,5]
    * exp002b: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_epochs in [-1,50,130,340,900],
      n_subjects in [1,2,3,4,5]
* exp003 (hyperparameter search)
    * exp003a: pre-training + downstream training with frozen FE, exploration of various pretraining hyperparameters
    * exp003b: pre-training + downstream training with finetuning of the FE, exploration of various pretraining
      hyperparameters

# Run Descriptions

This is basically the folder structure of the `logs/` directory.

* exp001
    * exp001a
        * sweep-2023-10-13_14-21-17: training with 3 different seeds --> 3*6*5=90 "runs" in total
        * sweep-2023-12-19_12-39-56: evaluation of sweep-2023-10-13_14-21-17 on the test folds
    * exp001b
        * 2023-11-28_14-13-48: single pretraining run with files containing the ground truth frequency bins and the
          predicted frequency bins after the last pretraining epoch on the validation set
        * sweep-2023-10-13_16-21-12: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2023-10-13_17-18-36: downstream training based on the pretrained models of
          exp001b/sweep-2023-10-13_16-21-12; 3 different seeds
        * sweep-2023-12-19_12-47-46: evaluation of sweep-2023-10-13_17-18-36 on the test folds
    * exp001c
        * sweep-2023-10-13_17-19-04: downstream training based on the pretrained models of
          exp001b/sweep-2023-10-13_16-21-12; 3 different seeds
        * sweep-2023-12-19_12-55-43: evaluation of sweep-2023-10-13_17-19-04 on the test folds
    * exp001d
        * sweep-2023-10-13_16-19-02: training with 3 different seeds
        * sweep-2023-12-19_13-03-30: evaluation of sweep-2023-10-13_16-19-02 on the test folds
* exp002
    * exp002a
        * sweep-2023-10-17_13-13-53: training with 3 different seeds --> 3*5*5*5=375 "runs" in total
        * sweep-2023-12-19_13-19-23: evaluation of sweep-2023-10-17_13-13-53 on the test folds
    * exp002b
        * sweep-2023-10-17_12-41-35: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2023-10-17_13-14-37: downstream training based on the pretrained models of
          exp002b/sweep-2023-10-17_12-41-35; 3 different seeds
        * sweep-2023-12-19_13-51-00: evaluation of sweep-2023-10-17_13-14-37 on the test folds
* exp003
    * exp003a
        * sweep-2023-11-10_19-52-30: search for number of frequencies in [5,10,15,20,30], 3 seeds
        * sweep-2023-11-11_02-19-07: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds
        * sweep-2023-11-11_13-09-09: exploration of logarithmic frequency bins, 3 seeds
    * exp003b
        * sweep-2023-11-10_19-52-44: search for number of frequencies in [5,10,15,20,30], 3 seeds
        * sweep-2023-11-11_03-11-14: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds
        * sweep-2023-11-11_14-36-23: exploration of logarithmic frequency bins, 3 seeds
