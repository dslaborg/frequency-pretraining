# exp005

This file describes how to reproduce the results presented in Figure 2 and 3 in our paper.
These results stem from four different training setups, which are grouped together in the experiment `exp005` as
sub-experiments `exp005a`, `exp005b`, `exp005c`, and `exp005d`.

- `exp005a`: Fully Supervised
- `exp005b`: Fixed Feature Extractor
- `exp005c`: Fine-Tuned Feature Extractor
- `exp005d`: Untrained Feature Extractor

In all training setups, a 5-fold cross-validation is performed.
To simplify the training code, cross-validation is not performed within the python code, but rather through separate
training runs started from the command line.
The cross-validation scheme is repeated three times with different seeds (i.e., in total 15 training runs for each
sub-experiment).
Each training run uses a different pretrained model, but the pretrained models are shared between training setups (i.e.,
15 models are pretrained in the pretraining phase).

## pretraining

We only need to perform pretraining for one training setup (either exp005b or exp005c), since the pretrained models are
shared between training setups.

```bash
# number of pretraining runs with different seeds
n_runs=15
# seeds are just an "array" of integers, e.g., [0,0,0,0,0],[1,1,1,1,1],...
seeds=''
for i in $(seq 0 $((n_runs-1))); do seeds+="[$i,$i,$i,$i,$i],"; done
seeds=${seeds::-1}
# start pretraining as a multirun with 15 runs
# the number of gpus and the number of parallel jobs can be adjusted to the available resources
python scripts/pretrain.py -cn=exp005/exp005b -m seeds="$seeds" general.gpus=[0] hydra.launcher.n_jobs=10
```

## fine-tuning

After pretraining, we can start the fine-tuning phase for all four training setups.
The script below needs to be modified slightly to match the actual model paths, which are dependent on the timestamp of
the pretraining runs.
You can find the actual model paths in the logs of the pretraining runs.
You only need to modify the `<timestamp>` placeholder in the script below.

```bash
# base model path points to the pretrained models from the pretraining run
# Since the model checkpoints include the timestamp of the pretraining run, the actual model paths need to be adjusted manually (e.g., exp005b-m###-simple_multi_class-2023-10-12_18-34-48-final.pth).
# the "###" is a placeholder for the index of the pretraining run (DO NOT MODIFY)
base_model_path='exp005b-m###-simple_multi_class-<timestamp>-final.pth'
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
    subject_ids="\${data.sleepedfx.cv_5_fold.fold_${j}}"
    # create a dict that will be used to overwrite the m_seed_path_sids parameter in the config
    m_seed_path_sids+="{seeds:${seed},path:\"$model_path\",subject_ids:{sleepedfx:${subject_ids}}},"
  done
done
m_seed_path_sids="${m_seed_path_sids::-1}"
# m_seed_path_sids should look like this: {seeds:[0,0,0,0,0],path:"exp005b-m0-simple_multi_class-2023-10-12_18-34-48-final.pth",subject_ids:{sleepedfx:${data.sleepedfx.cv_5_fold.fold_1}}},{seeds:[1,1,1,1,1],path:"exp005b-m1-simple_multi_class-2023-10-12_18-34-48-final.pth",subject_ids:{sleepedfx:${data.sleepedfx.cv_5_fold.fold_2}}},...

# start the fine-tuning phase as multiruns with 3 * 5 * 5 * 5 = 375 runs (number of repetitions * number of folds * number of n_subject values * number of n_epochs values)
# the number of gpus and the number of parallel jobs can be adjusted to the available resources
python scripts/fine-tune.py -cn=exp005/exp005a -m m_seed_path_sids="$m_seed_path_sids" data.downstream.train_dataloader.dataset.data_reducer.n_epochs=-1,50,130,340,900 data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1,2,3,4,5 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/fine-tune.py -cn=exp005/exp005b -m m_seed_path_sids="$m_seed_path_sids" data.downstream.train_dataloader.dataset.data_reducer.n_epochs=-1,50,130,340,900 data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1,2,3,4,5 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/fine-tune.py -cn=exp005/exp005c -m m_seed_path_sids="$m_seed_path_sids" data.downstream.train_dataloader.dataset.data_reducer.n_epochs=-1,50,130,340,900 data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1,2,3,4,5 general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/fine-tune.py -cn=exp005/exp005d -m m_seed_path_sids="$m_seed_path_sids" data.downstream.train_dataloader.dataset.data_reducer.n_epochs=-1,50,130,340,900 data.downstream.train_dataloader.dataset.data_reducer.n_subjects=1,2,3,4,5 general.gpus=[0] hydra.launcher.n_jobs=10
```

## evaluation on test set

During the fine-tuning phase, models are only evaluated on the validation data for early stopping.
After the fine-tuning phase, we can then evaluate the best model for each fold of the cross-validation scheme on the
test set.
The script below needs to be modified slightly to match the actual model paths, which are dependent on the timestamp of
the fine-tuning runs.
You can find the timestamps of the fine-tuning runs in their logs or in the folder names of the result folders.
You only need to modify the `<timestamp>` placeholders in the script below.

```bash
# timestamps of the fine-tuning runs (e.g., 2023-10-13_14-21-17)
# since each sub-experiment is fine-tuned separately, the timestamps are different for each sub-experiment
exp005a_run='<timestamp>'
exp005b_run='<timestamp>'
exp005c_run='<timestamp>'
exp005d_run='<timestamp>'

# base model path points to the best models from the fine-tuning phase
# placeholders, that are replaced by the actual values in the loop (DO NOT MODIFY):
# #exp#: sub-experiment name (exp005a, exp005b, exp005c, exp005d)
# #run_index#: index of the fine-tuning run
# #run_name#: timestamp of the fine-tuning run
base_model_path='#exp#-m#run_index#-base_fe_clas-#run_name#-final.pth'

# number of repetitions of the cross-validation scheme
n_runs=3
# number of folds in the cross-validation scheme
n_folds=5
# number of data reductions that were explored during the fine-tuning phase
n_data_reductions=25  # 5 n_subjects values * 5 n_epochs values
m_seed_path_sids=''
i_run=0
for i in $(seq 0 $((n_runs-1))); do
  for j in $(seq 1 $((n_folds))); do
    for k in $(seq 1 $((n_data_reductions))); do
      # replace the run_index placeholder with the actual index of the fine-tuning run
      model_path=${base_model_path//#run_index#/$((i_run))}
      # specify the fold of the cross-validation scheme
      subject_ids="\${data.sleepedfx.cv_5_fold.fold_${j}}"
      # create a dict that will be used to overwrite the m_seed_path_sids parameter in the config
      m_seed_path_sids+="{path:\"$model_path\",subject_ids:{sleepedfx:${subject_ids}}},"
      # increment the run_index for each repetition, fold, and data reduction
      i_run=$((i_run+1))
    done
  done
done
m_seed_path_sids="${m_seed_path_sids::-1}"
# m_seed_path_sids should look like this: {path:"#exp#-m0-base_fe_clas-#run_name#-final.pth",subject_ids:{sleepedfx:${data.sleepedfx.cv_5_fold.fold_1}}},{path:"#exp#-m1-base_fe_clas-#run_name#-final.pth",subject_ids:{sleepedfx:${data.sleepedfx.cv_5_fold.fold_2}}},...

# replace the #exp# and #run_name# placeholders with the actual values for each sub-experiment
m_seed_path_sids_005a=${m_seed_path_sids//#exp#/exp005a}
m_seed_path_sids_005a=${m_seed_path_sids_005a//#run_name#/$exp005a_run}
m_seed_path_sids_005b=${m_seed_path_sids//#exp#/exp005b}
m_seed_path_sids_005b=${m_seed_path_sids_005b//#run_name#/$exp005b_run}
m_seed_path_sids_005c=${m_seed_path_sids//#exp#/exp005c}
m_seed_path_sids_005c=${m_seed_path_sids_005c//#run_name#/$exp005c_run}
m_seed_path_sids_005d=${m_seed_path_sids//#exp#/exp005d}
m_seed_path_sids_005d=${m_seed_path_sids_005d//#run_name#/$exp005d_run}

# start the evaluation on the test set as multiruns with 3 * 5 * 5 * 5 = 375 runs (number of repetitions * number of folds * number of n_subject values * number of n_epochs values)
# the number of gpus and the number of parallel jobs can be adjusted to the available resources
# the model.downstream.path parameter is added to point towards the model path defined in m_seed_path_sids
# since we always load the full model, the model.downstream.feature_extractor.path parameter is set to null
# the training.downstream.trainer.evaluators.test parameter is added to specify the evaluator that should be used for the test set
python scripts/eval_fine-tuned.py -cn=exp005/exp005a -m m_seed_path_sids="$m_seed_path_sids_005a" +model.downstream.path='${m_seed_path_sids.path}' +training.downstream.trainer.evaluators.test='${evaluators.downstream.test}' model.downstream.feature_extractor.path=null general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/eval_fine-tuned.py -cn=exp005/exp005b -m m_seed_path_sids="$m_seed_path_sids_005b" +model.downstream.path='${m_seed_path_sids.path}' +training.downstream.trainer.evaluators.test='${evaluators.downstream.test}' model.downstream.feature_extractor.path=null general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/eval_fine-tuned.py -cn=exp005/exp005c -m m_seed_path_sids="$m_seed_path_sids_005c" +model.downstream.path='${m_seed_path_sids.path}' +training.downstream.trainer.evaluators.test='${evaluators.downstream.test}' model.downstream.feature_extractor.path=null general.gpus=[0] hydra.launcher.n_jobs=10
python scripts/eval_fine-tuned.py -cn=exp005/exp005d -m m_seed_path_sids="$m_seed_path_sids_005d" +model.downstream.path='${m_seed_path_sids.path}' +training.downstream.trainer.evaluators.test='${evaluators.downstream.test}' model.downstream.feature_extractor.path=null general.gpus=[0] hydra.launcher.n_jobs=10
```
