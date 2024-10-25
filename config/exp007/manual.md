# exp007

Experiment exp007 explores various hyperparameters around the pretraining process.
There are two different training setups, which are grouped together in the experiment `exp007` as
sub-experiments `exp007a` and `exp007b`.

- `exp007a`: Fixed Feature Extractor
- `exp007b`: Fine-Tuned Feature Extractor

In all training setups, a 5-fold cross-validation is performed.
To simplify the training code, cross-validation is not performed within the python code, but rather through separate
training runs started from the command line.
The cross-validation scheme is repeated three times with different seeds (i.e., in total 15 training runs for each
sub-experiment).
Different from the previous experiments, the pretrained models are not shared between training setups, but rather
trained separately for each training setup.

## training

Pretraining and fine-tuning are combined into a single script.
The code block below describes three different hyperparameter explorations for each training setup:

1. Different numbers of frequency bins
2. Different numbers of samples in the training set during pretraining
3. Logarithmic frequency bins vs. linear frequency bins
4. Explore variety of pretraining samples; number of samples is fixed to 100000 by under-/oversampling; downstream
   training on one subject

```bash
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
    # define the fold of the cross-validation scheme
    subject_ids="\${data.dod_o_h.cv_5_fold.fold_${j}}"
    # create a dict that will be used to overwrite the m_seed_path_sids parameter in the config
    m_seed_path_sids+="{seeds:${seed},subject_ids:{dod_o_h:${subject_ids}}},"
  done
done
m_seed_path_sids="${m_seed_path_sids::-1}"
# m_seed_path_sids should look like this: {seeds:[0,0,0,0,0],subject_ids:{dod_o_h:${data.dod_o_h.cv_5_fold.fold_1}}},{seeds:[1,1,1,1,1],subject_ids:{dod_o_h:${data.dod_o_h.cv_5_fold.fold_2}}},...

# start pretraining and fine-tuning as multiruns with 3 * 5 * x runs (number of repetitions * number of folds * number of hyperparameters to explore)
# each run consists of both pretraining and fine-tuning
# the number of gpus and the number of parallel jobs can be adjusted to the available resources

# hyperparameter exploration 1: different numbers of frequency bins
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.n_freqs=5,10,15,20,30 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.n_freqs=5,10,15,20,30 general.gpus=[0] hydra.launcher.n_jobs=10

# hyperparameter exploration 2: different numbers of samples in the training set
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.train_dataloader.dataset.n_samples=1000,10000,100000,1000000 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.train_dataloader.dataset.n_samples=1000,10000,100000,1000000 general.gpus=[0] hydra.launcher.n_jobs=10

# hyperparameter exploration 3: logarithmic frequency bins vs. linear frequency bins
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007a -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.log_bins=true,false general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007b -m m_seed_path_sids="$m_seed_path_sids" data.pretraining.log_bins=true,false general.gpus=[0] hydra.launcher.n_jobs=10

# hyperparameter exploration 4: explore variety of pretraining samples; number of samples is fixed to 100000 by under-/oversampling; downstream training on one subject
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007a -m \
  m_seed_path_sids="$m_seed_path_sids" \
  data.pretraining.train_dataloader.dataset.n_samples=1,10,100,1000,10000,100000,1000000 \
  +data.pretraining.train_dataloader.dataset.n_samples_per_epoch=100000 \
  data.downstream.train_dataloader.dataset.data_reducer._target_=base.data.data_reducer.SubjectWiseDataReducer \
  +data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1 \
  general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/pretrain_and_fine-tune.py -cn=exp007/exp007b -m \
  m_seed_path_sids="$m_seed_path_sids" \
  data.pretraining.train_dataloader.dataset.n_samples=1,10,100,1000,10000,100000,1000000 \
  +data.pretraining.train_dataloader.dataset.n_samples_per_epoch=100000 \
  data.downstream.train_dataloader.dataset.data_reducer._target_=base.data.data_reducer.SubjectWiseDataReducer \
  +data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1 \
  general.gpus=[0] hydra.launcher.n_jobs=10
```
