# Experiment Descriptions

* exp001 (DODO/H datasets; full data)
    * exp001a: fully supervised training, full training data; 3 different seeds
    * exp001b: pre-training + downstream training with fixed FE, full training data; 3 different seeds
    * exp001c: pre-training + downstream training with finetuning of the FE, full training data; 3 different seeds
    * exp001d: no pre-training + downstream training with frozen FE, full training data; 3 different seeds
* exp002 (Sleep-EDFx dataset; full data)
    * exp002a: fully supervised training, full training data; 3 different seeds
    * exp002b: pre-training + downstream training with fixed FE, full training data; 3 different seeds
    * exp002c: pre-training + downstream training with finetuning of the FE, full training data; 3 different seeds
    * exp002d: no pre-training + downstream training with frozen FE, full training data; 3 different seeds
* exp003 (ISRUC dataset; full data)
    * exp003a: fully supervised training, full training data; 3 different seeds
    * exp003b: pre-training + downstream training with fixed FE, full training data; 3 different seeds
    * exp003c: pre-training + downstream training with finetuning of the FE, full training data; 3 different seeds
    * exp003d: no pre-training + downstream training with frozen FE, full training data; 3 different seeds
* exp004 (DODO/H datasets; explore number of epochs and subjects)
    * exp004a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (
      each with 3 different seeds): n_epochs in [-1,50,130,340,900], n_subjects in [1,2,3,4,5]
    * exp004b: pre-training + downstream training with fixed FE, training in the lowest data regime (n_subjects=1 and
      n_epochs=50); 3 different seeds
    * exp004c: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_epochs in [-1,50,130,340,900],
      n_subjects in [1,2,3,4,5]
    * exp004d: no pre-training + downstream training with frozen FE, training in the lowest data regime (n_subjects=1
      and n_epochs=50); 3 different seeds
* exp005 (Sleep-EDFx dataset; explore number of epochs and subjects)
    * exp005a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (
      each with 3 different seeds): n_epochs in [-1,50,130,340,900], n_subjects in [1,2,3,4,5]
    * exp005b: pre-training + downstream training with fixed FE, training in the lowest data regime (n_subjects=1
      and n_epochs=50); 3 different seeds
    * exp005c: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_epochs in [-1,50,130,340,900],
      n_subjects in [1,2,3,4,5]
    * exp005d: no pre-training + downstream training with frozen FE, training in the lowest data regime (n_subjects=1
      and n_epochs=50); 3 different seeds
* exp006 (ISRUC dataset; explore number of epochs and subjects)
    * exp006a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (
      each with 3 different seeds): n_epochs in [-1,50,130,340,900], n_subjects in [1,2,3,4,5]
    * exp006b: pre-training + downstream training with fixed FE, training in the lowest data regime (n_subjects=1
      and n_epochs=50); 3 different seeds
    * exp006c: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_epochs in [-1,50,130,340,900],
      n_subjects in [1,2,3,4,5]
    * exp006d: no pre-training + downstream training with frozen FE, training in the lowest data regime (n_subjects=1
      and n_epochs=50); 3 different seeds
* exp007 (hyperparameter search)
    * exp007a: pre-training + downstream training with frozen FE, exploration of various pretraining hyperparameters
    * exp007b: pre-training + downstream training with finetuning of the FE, exploration of various pretraining
      hyperparameters

# Run Descriptions

Folder structure of the `logs/` directory:

* exp001
    * exp001a
        * sweep-2023-10-13_14-21-17: training with 3 different seeds for 6 different subject numbers (we only need the
          full subject number) --> 3*6*5=90 "runs" in total
        * sweep-2023-12-19_12-39-56: evaluation of sweep-2023-10-13_14-21-17 on the test folds
    * exp001b
        * 2023-11-28_14-13-48: single pretraining run with files containing the ground truth frequency bins and the
          predicted frequency bins after the last pretraining epoch on the validation set
        * sweep-2023-10-13_16-21-12: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2023-10-13_17-18-36: downstream training based on the pretrained models of
          exp001b/sweep-2023-10-13_16-21-12; 3 different seeds; 6 subject numbers
        * sweep-2023-12-19_12-47-46: evaluation of sweep-2023-10-13_17-18-36 on the test folds
    * exp001c
        * sweep-2023-10-13_17-19-04: downstream training based on the pretrained models of
          exp001b/sweep-2023-10-13_16-21-12; 3 different seeds; 6 subject numbers
        * sweep-2023-12-19_12-55-43: evaluation of sweep-2023-10-13_17-19-04 on the test folds
    * exp001d
        * sweep-2023-10-13_16-19-02: training with 3 different seeds for 6 different subject numbers (we only need the
          full subject number) --> 3*6*5=90 "runs" in total
        * sweep-2023-12-19_13-03-30: evaluation of sweep-2023-10-13_16-19-02 on the test folds
* exp002
    * exp002a
        * sweep-2024-10-01_15-42-22: training with 3 different seeds -> 3*5=15 "runs" in total
        * sweep-2024-10-02_10-32-20: evaluation of sweep-2024-10-01_15-42-22 on the test folds
    * exp002b
        * sweep-2024-09-16_15-54-06: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2024-10-01_18-43-17: downstream training based on the pretrained models of
          exp002b/sweep-2024-09-16_15-54-06; 3 different seeds
        * sweep-2024-10-02_10-34-47: evaluation of sweep-2024-10-01_18-43-17 on the test folds
    * exp002c
        * sweep-2024-10-01_22-29-00: downstream training based on the pretrained models of
          exp002b/sweep-2024-09-16_15-54-06; 3 different seeds
        * sweep-2024-10-02_10-37-12: evaluation of sweep-2024-10-01_22-29-00 on the test folds
    * exp002d
        * sweep-2024-10-02_01-06-31: training with 3 different seeds -> 3*5=15 "runs" in total
        * sweep-2024-10-02_10-39-29: evaluation of sweep-2024-10-02_01-06-31 on the test folds
* exp003
    * exp003a
        * sweep-2024-10-01_15-44-21: training with 3 different seeds -> 3*5=15 "runs" in total
        * sweep-2024-10-02_10-32-24: evaluation of sweep-2024-10-01_15-44-21 on the test folds
    * exp003b
        * sweep-2024-09-17_19-35-00: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2024-10-01_18-25-45: downstream training based on the pretrained models of
          exp003b/sweep-2024-09-17_19-35-00; 3 different seeds
        * sweep-2024-10-02_10-33-57: evaluation of sweep-2024-10-01_18-25-45 on the test folds
    * exp003c
        * sweep-2024-10-01_20-38-01: downstream training based on the pretrained models of
          exp003b/sweep-2024-09-17_19-35-00; 3 different seeds
        * sweep-2024-10-02_10-35-22: evaluation of sweep-2024-10-01_20-38-01 on the test folds
    * exp003d
        * sweep-2024-10-01_22-37-40: training with 3 different seeds -> 3*5=15 "runs" in total
        * sweep-2024-10-02_10-36-54: evaluation of sweep-2024-10-01_22-37-40 on the test folds
* exp004
    * exp004a
        * sweep-2023-10-17_13-13-53: training with 3 different seeds --> 3*5*5*5=375 "runs" in total
        * sweep-2023-12-19_13-19-23: evaluation of sweep-2023-10-17_13-13-53 on the test folds
    * exp004b
        * sweep-2023-10-17_12-41-35: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2024-10-07_19-02-25: downstream training based on the pretrained models of
          exp004b/sweep-2023-10-17_12-41-35; 3 different seeds
        * sweep-2024-10-08_09-43-29: evaluation of sweep-2024-10-07_19-02-25 on the test folds
    * exp004c
        * sweep-2023-10-17_13-14-37: downstream training based on the pretrained models of
          exp004b/sweep-2023-10-17_12-41-35; 3 different seeds
        * sweep-2023-12-19_13-51-00: evaluation of sweep-2023-10-17_13-14-37 on the test folds
    * exp004d
        * sweep-2024-10-07_20-25-29: training with 3 different seeds --> 3*5=15 "runs" in total
        * sweep-2024-10-08_09-44-30: evaluation of sweep-2024-10-07_20-25-29 on the test folds
* exp005
    * exp005a
        * sweep-2024-10-02_10-52-09: training with 3 different seeds --> 3*5*5*5=375 "runs" in total
        * sweep-2024-10-05_00-50-56: evaluation of sweep-2024-10-02_10-52-09 on the test folds
    * exp005b
        * sweep-2024-09-20_13-57-42: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2024-09-20_22-20-56: downstream training based on the pretrained models of
          exp005b/sweep-2024-09-20_13-57-42; 3 different seeds
        * sweep-2024-09-23_09-49-10: evaluation of sweep-2024-09-20_22-20-56 on the test folds
    * exp005c
        * sweep-2024-10-04_03-09-39: downstream training based on the pretrained models of
          exp005b/sweep-2024-09-20_13-57-42; 3 different seeds
        * sweep-2024-10-05_01-17-51: evaluation of sweep-2024-10-04_03-09-39 on the test folds
    * exp005d
        * sweep-2024-09-20_20-46-42: training with 3 different seeds --> 3*5=15 "runs" in total
        * sweep-2024-09-23_10-01-09: evaluation of sweep-2024-09-20_20-46-42 on the test folds
* exp006
    * exp006a
        * sweep-2024-10-02_10-52-14: training with 3 different seeds --> 3*5*5*5=375 "runs" in total
        * sweep-2024-10-04_15-33-59: evaluation of sweep-2024-10-02_10-52-14 on the test folds
    * exp006b
        * sweep-2024-09-20_13-58-02: pretraining for 15 different seeds (different models for each seed and fold)
        * sweep-2024-09-20_20-27-59: downstream training based on the pretrained models of
          exp006b/sweep-2024-09-20_13-58-02; 3 different seeds
        * sweep-2024-09-23_09-48-41: evaluation of sweep-2024-09-20_20-27-59 on the test folds
    * exp006c
        * sweep-2024-10-03_20-03-43: downstream training based on the pretrained models of
          exp006b/sweep-2024-09-20_13-58-02; 3 different seeds
        * sweep-2024-10-04_16-00-41: evaluation of sweep-2024-10-03_20-03-43 on the test folds
    * exp006d
        * sweep-2024-09-20_19-02-44: training with 3 different seeds --> 3*5=15 "runs" in total
        * sweep-2024-09-23_09-56-13: evaluation of sweep-2024-09-20_19-02-44 on the test folds
* exp007
    * exp007a
        * sweep-2023-11-10_19-52-30: search for number of frequencies in [5,10,15,20,30], 3 seeds
        * sweep-2023-11-11_02-19-07: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds
        * sweep-2023-11-11_13-09-09: exploration of logarithmic frequency bins, 3 seeds
        * sweep-2024-09-11_16-54-46: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds,
          over-/undersampling to 100,000 samples, 1 random subject in downstream training
        * sweep-2024-09-12_12-09-21: search for number of pretraining samples in [1,10,100], 3 seeds,
          over-/undersampling to 100,000 samples, 1 random subject in downstream training
    * exp007b
        * sweep-2023-11-10_19-52-44: search for number of frequencies in [5,10,15,20,30], 3 seeds
        * sweep-2023-11-11_03-11-14: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds
        * sweep-2023-11-11_14-36-23: exploration of logarithmic frequency bins, 3 seeds
        * sweep-2024-09-11_16-55-09: search for number of pretraining samples in [1000,10000,100000,1000000], 3 seeds,
          over-/undersampling to 100,000 samples, 1 random subject in downstream training
        * sweep-2024-09-12_13-51-12: search for number of pretraining samples in [1,10,100], 3 seeds,
          over-/undersampling to 100,000 samples, 1 random subject in downstream training
