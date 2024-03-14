# exp003

Experiment exp003 explores various hyperparameters around the pretraining process.
There are two different training setups, which are grouped together in the experiment `exp003` as
sub-experiments `exp003a` and `exp003b`.

- `exp003a`: Fixed Feature Extractor
- `exp003b`: Fine-Tuned Feature Extractor

In all training setups, a 5-fold cross-validation is performed.
To simplify the training code, cross-validation is not performed within the python code, but rather through separate
training runs started from the command line.
The cross-validation scheme is repeated three times with different seeds (i.e., in total 15 training runs for each
sub-experiment).
Different from exp001 and exp003, the pretrained models are not shared between training setups, but rather trained
separately for each training setup.

## training

Pretraining and fine-tuning are combined into a single script.
The code block below describes three different hyperparameter explorations for each training setup:

1. Different numbers of frequency bins
2. Different numbers of samples in the training set
3. Logarithmic frequency bins vs. linear frequency bins

```bash
# base model path points to the pretrained models from the pretraining run
# Since the model checkpoints include the timestamp of the pretraining run, the actual model paths need to be adjusted manually (e.g., exp003b-m###-simple_multi_class-2023-10-12_18-34-48-final.pth).
# the "###" is a placeholder for the index of the pretraining run (DO NOT MODIFY)
base_model_path='exp003b-m###-simple_multi_class-<timestamp>-final.pth'
# number of repetitions of the cross-validation scheme
n_runs=3
# number of folds in the cross-validation scheme
n_folds=5
m_seed_path_sids=''
for i in $(seq 0 $((n_runs-1))); do
  for j in $(seq 1 $((n_folds))); do
    # use a different seed for each repetition and fold of the cross-validation scheme
    seed_base=$((i*n_folds+j-1))
    seed="[${seed_base},${seed_base},${seed_base},${seed_base},${seed_base}]"
    # replace the placeholder with the actual index of the pretraining run, which is the same as the seed
    model_path=${base_model_path//###/${seed_base}}
    # define the fold of the cross-validation scheme
    subject_ids="\${data.dod_o_h.cv_5_fold.fold_${j}}"
    # create a dict that will be used to overwrite the m_seed_path_sids parameter in the config
    m_seed_path_sids+="{seeds:${seed},path:\"$model_path\",subject_ids:{dod_o_h:${subject_ids}}},"
  done
done
m_seed_path_sids="${m_seed_path_sids::-1}"
# m_seed_path_sids should look like this: {seeds:[0,0,0,0,0],path:"exp003b-m0-simple_multi_class-2023-10-12_18-34-48-final.pth",subject_ids:{dod_o_h:${data.dod_o_h.cv_5_fold.fold_1}}},{seeds:[1,1,1,1,1],path:"exp003b-m1-simple_multi_class-2023-10-12_18-34-48-final.pth",subject_ids:{dod_o_h:${data.dod_o_h.cv_5_fold.fold_2}}},...

# start pretraining and fine-tuning as multiruns with 3 * 5 * x runs (number of repetitions * number of folds * number of hyperparameters to explore)
# each run consists of both pretraining and fine-tuning
# the number of gpus and the number of parallel jobs can be adjusted to the available resources

# hyperparameter exploration 1: different numbers of frequency bins
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.n_freqs=5,10,15,20,30 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.n_freqs=5,10,15,20,30 general.gpus=[0] hydra.launcher.n_jobs=10

# hyperparameter exploration 2: different numbers of samples in the training set
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.train_dataloader.dataset.n_samples=1000,10000,100000,1000000 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.train_dataloader.dataset.n_samples=1000,10000,100000,1000000 general.gpus=[0] hydra.launcher.n_jobs=10

# hyperparameter exploration 3: logarithmic frequency bins vs. linear frequency bins
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.log_bins=true,false general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp003/exp003b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.log_bins=true,false general.gpus=[0] hydra.launcher.n_jobs=10
```
