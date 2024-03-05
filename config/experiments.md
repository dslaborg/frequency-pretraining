# Experiment Descriptions

* exp001
    * exp001a: fully supervised training, a Gridsearch over the following hyperparameter combinations is performed (each
      with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001b: pre-training + downstream training with frozen FE, a Gridsearch over the following hyperparameter
      combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001c: pre-training + downstream training with finetuning of the FE, a Gridsearch over the following
      hyperparameter combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]
    * exp001d: no pre-training + downstream training with frozen FE, a Gridsearch over the following hyperparameter
      combinations is performed (each with 3 different seeds): n_subjects in [54, 25, 10, 5, 2, 1]

# Run Descriptions

This is basically the folder structure of the `logs/` directory.

* exp001
    * exp001a
        * sweep-2023-10-13_14-21-17: training with 3 different seeds --> 3*6*5=90 "runs" in total
        * sweep-2023-12-19_12-39-56: evaluation of sweep-2023-10-13_14-21-17 on the test folds
    * exp001b
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
