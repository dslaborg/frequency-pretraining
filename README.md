# Frequency Pretraining (FPT)

## About The Project

Implementation of Frequency Pretraining (FPT) as described in:

> Niklas Grieger, Siamak Mehrkanoon and Stephan Bialonski. Data-Efficient Sleep Staging with Synthetic Time Series
> Pretraining. arXiv:2403.08592, 2021. URL https://arxiv.org/abs/2403.08592

Part of our work was presented at the ICLR 2024 Workshop on Learning from Time Series for Health (the code for this
version of the paper can be found in the `ts4h-ext-abstract-version` branch):

> Niklas Grieger, Siamak Mehrkanoon and Stephan Bialonski. Pretraining Sleep Staging Models without Patient Data. In
*ICLR 2024 Workshop on Learning from Time Series for Health*, Vienna, Austria, 2024.
> URL https://openreview.net/forum?id=xOchS6sthY

## Project Structure

The project is set up as follows:

* `base/`: contains the python implementations of the FPT scheme
* `cache/dod_o_h/`: contains the preprocessed data, as produced by the `prepare_dodh.py` and `prepare_dodo.py` scripts
* `config/`: contains the configurations of the experiments, configuring how to train or evaluate the model;
  configurations are based on the [hydra](https://hydra.cc) framework
    * `data/`: base configurations around dataloading and data splits
    * `exp001/`: experiments are groups of sub-experiments, e.g. `exp001` contains the four
      sub-experiments `exp001a`, `exp001b`, `exp001c`, and `exp001d`
        * `exp001a.yaml`, `exp001b.yaml`, `exp001c.yaml`, `exp001d.yaml`: configurations for the four sub-experiments
        * `fpt_config.yaml`: base configuration for the `exp001` experiment group
        * `manual.md`: manual for the `exp001` experiment group, which describes how to run the experiments and evaluate
          the models; the manual is used to reproduce the results of the paper
    * `exp002/`, ...: similar to `exp001`, but for a different set of experiments
    * `launcher/`: base config around the launcher that is used by hydra to launch runs
    * `sweeper/`: base config around the sweeper that is used by hydra to sweep over parameter ranges in multiruns
    * `base_config.yaml`: base configuration for all experiments; describes the rough outline of experiment
      configurations
    * `experiments.md`: describes the existing experiments and where to find the results of training and evaluation
* `logs/`: contains the logs of the training and evaluation runs; the folder structure is described in the
  `config/experiments.md` file
* `models/`: contains the trained model checkpoints
* `preprocessing/`: contains scripts used to preprocess the data
* `scripts/`: contains training and evaluation scripts, which are used for model training and subsequent evaluation in
  experiments (as configured within the `config` directory); also contains scripts used to create the figures in our
  paper

## Installation

On Linux and Windows the project can be used by running the following commands to clone the repository and install the
required dependencies.

### Either with `anaconda` or `miniconda` installed (recommended)

```shell
git clone https://github.com/dslaborg/frequency-pretraining.git
cd frequency-pretraining
conda create -n fpt python=3.10
conda activate fpt

# install pytorch following the instructions at https://pytorch.org/get-started/locally/
# e.g. for CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# install remaining dependencies
pip install -r requirements.txt
```

### or using `pip`

```shell
git clone https://github.com/dslaborg/frequency-pretraining.git
cd frequency-pretraining

# install pytorch following the instructions at https://pytorch.org/get-started/locally/
# e.g. for the latest CUDA version
pip install torch

# install remaining dependencies
pip install -r requirements.txt
```

## Reproducing the results of the paper

To reproduce the results described in the paper, you need to (i) download the data, (ii) preprocess the data, (iii)
pretrain/fine-tune the models according to exp001a-exp001d, and (iv) evaluate the models:

1. Download the data:
    - DODO/H datasets:
        1. Download the EEG signals using the [download_data.py](preprocessing/dreem/download_data.py) script.
        2. Download the annotations from
           the [dreem-learning-evaluation](https://github.com/Dreem-Organization/dreem-learning-evaluation) repository.
    - Sleep-EDFx dataset:
        1. Download the data from the [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) website.
    - ISRUC dataset:
        1. Download the data from [their website](https://sleeptight.isr.uc.pt/).
2. Preprocessing of the data:
    - DODO/H datasets:
        1. Preprocess the data using the [prepare_dodh.py](preprocessing/dreem/prepare_dodh.py)
           and [prepare_dodo.py](preprocessing/dreem/prepare_dodo.py) scripts (see below for details on the parameters
           of the scripts).
        2. Copy all preprocessed datafiles to the `cache/dod_o_h` directory.
    - Sleep-EDFx dataset:
        1. Preprocess the data using the [prepare_sleepedfx.py](preprocessing/sleepedfx/prepare_sleepedfx.py)
           script (see below for details on the parameters of the script).
    - ISRUC dataset:
        1. Preprocess the data using the [prepare_isruc.py](preprocessing/isruc/prepare_isruc.py) script (see below
           for details on the parameters of the script).
3. Follow the instructions given in the manual files of the experiments to pretrain and fine-tune the models.
    1. For Figure 2, you need the results of exp001-exp006. The manual for exp001 can be found at
       [config/exp001/manual.md](config/exp001/manual.md). The manuals of other experiments follow the same structure.
    2. For Figure 3, you need the results of exp004-exp006. The manual for exp004 can be found at
       [config/exp004/manual.md](config/exp004/manual.md). The manuals of other experiments follow the same structure.
    3. For Figure 4, you need the results of exp001 and exp007. The manual for exp007 can be found at
       [config/exp007/manual.md](config/exp007/manual.md).
4. Evaluation on the test set is also described in the manual files.

## Scripts

### Preprocessing scripts

The `preprocessing` directory contains scripts used to preprocess the data (in our case, the `dreem` (DODO/H),
`sleepedfx`, and `isruc` datasets).

#### dreem/download_data.py

Downloads the EEG signals from the Dreem dataset to `~/data/dreem`.

Sample call: `python preprocessing/dreem/download_data.py`

#### dreem/prepare_dodh.py

Preprocessing script for the DODH dataset.

Arguments:

* `-s` or `--signals_dir`: path to the directory containing the EEG signals
* `-a` or `--annotations_dir`: path to the directory containing the annotations
* `-o` or `--output_dir`: path to the directory where the preprocessed data should be saved; default is `cache/dodh`

Sample call: `python preprocessing/dreem/prepare_dodh.py -s ~/data/dreem -a ~/data/dreem -o cache/dodh`

#### dreem/prepare_dodo.py

Preprocessing script for the DODO dataset.

Arguments:

* `-s` or `--signals_dir`: path to the directory containing the EEG signals
* `-a` or `--annotations_dir`: path to the directory containing the annotations
* `-o` or `--output_dir`: path to the directory where the preprocessed data should be saved; default is `cache/dodo`

Sample call: `python preprocessing/dreem/prepare_dodo.py -s ~/data/dreem -a ~/data/dreem -o cache/dodo`

#### sleepedfx/prepare_sleepedfx.py

Preprocessing script for the Sleep-EDFx dataset.

Arguments:

* `-d` or `--data_dir`: path to the directory containing the EEG signals (*PSG.edf) and the corresponding annotations
  (*Hypnogram.edf)
* `-o` or `--output_dir`: path to the directory where the preprocessed data should be saved; default is
  `cache/sleep-edfx`

Sample call:
`python preprocessing/sleepedfx/prepare_sleepedfx.py -d ~/data/sleepedfx -o cache/sleep-edfx`

#### isruc/prepare_isruc.py

Preprocessing script for the ISRUC dataset.

Arguments:

* `-s` or `--signals_dir`: path to the directory containing the folder structure from the ISRUC website with the EEG
  signals (.rec) and the corresponding annotations (.txt)
* `-o` or `--output_dir`: path to the directory where the preprocessed data should be saved; default is `cache/isruc`

Sample call: `python preprocessing/isruc/prepare_isruc.py -s ~/data/isruc -o cache/isruc`

### Running and evaluating an experiment

All training and evaluation scripts can be found in the `scripts` directory.
The scripts require configuration files, which are expected to be located in the `config` directory (see section "
Configuration" for details).

#### pretrain.py

Performs pretraining as specified in the corresponding configuration file, writes its log to the console and saves a log
file and results to a result directory in the `logs` directory.
Model checkpoints are written to the `models` directory.

Arguments:

* this is a hydra-based script, which means that any configuration can be overwritten using command line arguments (see
  section "Configuration" for details)
* `-m`: sets the script to the `multirun` mode (see section "Configuration" for details)
* `-cn=<experiment group>/<sub-experiment>`: name of experiment to run, for which a `<sub-experiment>.yaml` file has to
  exist in the `config/<experiment group>` directory

Sample call (single run): `python scripts/pretrain.py -cn=exp001/exp001b`

#### fine-tune.py

Performs fine-tuning as specified in the corresponding configuration file, writes its log to the console and saves a log
file and results to a result directory in the `logs` directory.
Model checkpoints are written to the `models` directory.

Arguments:

* this is a hydra-based script, which means that any configuration can be overwritten using command line arguments (see
  section "Configuration" for details)
* `-m`: sets the script to the `multirun` mode (see section "Configuration" for details)
* `-cn=<experiment group>/<sub-experiment>`: name of experiment to run, for which a `<sub-experiment>.yaml` file has to
  exist in the `config/<experiment group>` directory

Sample call (single run): `python scripts/fine-tune.py -cn=exp001/exp001a`

#### pretrain_and_fine-tune.py

First, performs pretraining, then fine-tuning as specified in the corresponding configuration file, writes its log to
the console and saves a log
file and results to a result directory in the `logs` directory.
Model checkpoints are written to the `models` directory.

Arguments:

* this is a hydra-based script, which means that any configuration can be overwritten using command line arguments (see
  section "Configuration" for details)
* `-m`: sets the script to the `multirun` mode (see section "Configuration" for details)
* `-cn=<experiment group>/<sub-experiment>`: name of experiment to run, for which a `<sub-experiment>.yaml` file has to
  exist in the `config/<experiment group>` directory

Sample call (single run): `python scripts/pretrain_and_fine-tune.py -cn=exp001/exp001b`

#### eval_fine-tuned.py

Evaluates a model as specified in the corresponding configuration file, writes its log to the console and saves a log
file and results to a result directory in the `logs` directory.

Arguments:

* this is a hydra-based script, which means that any configuration can be overwritten using command line arguments (see
  section "Configuration" for details)
* `-m`: sets the script to the `multirun` mode (see section "Configuration" for details)
* `-cn=<experiment group>/<sub-experiment>`: name of experiment to run, for which a `<sub-experiment>.yaml` file has to
  exist in the `config/<experiment group>` directory

Sample call (single
run):
`python scripts/eval_fine-tuned.py -cn=exp001/exp001a +model.downstream.path='exp001b-base_fe_clas-2023-10-13_14-21-17-final.pth' +training.downstream.trainer.evaluators.test='${evaluators.downstream.test}' model.downstream.feature_extractor.path=null`

Explanation of the sample call: The `+model.downstream.path` parameter specifies the path to the model checkpoint that
should
be evaluated.
The `+training.downstream.trainer.evaluators.test` parameter specifies the evaluator that should be used for evaluation.
In this case, we want to evaluate on the test set and use the test evaluator that was defined
in `exp001/fpt_config.yaml` under the key `evaluators.downstream.test`.
Since both the model path and the evaluator weren't part of the configuration before, we add them using the `+` prefix.
The last parameter `model.downstream.feature_extractor.path=null` is used to overwrite the feature extractor path, which
is not needed for evaluation because we always load the full model.

### Visualization scripts

The `scripts/visualization` directory contains scripts used to create the figures used in our paper.

#### visualize_matrix_nepochs_vs_nsubjects_testdata_cv.py

Creates the plot used in Figure 3 of the paper by reading the results of experiment groups exp004-exp006 from the `logs`
directory.

Sample call: `python scripts/visualization/visualize_matrix_nepochs_vs_nsubjects_testdata_cv.py`

#### visualize_metrics_pretraining.py

Creates the plot used in Figure 4 of the paper by reading the results of experiment groups exp001 and exp007 from the
`logs` directory.

Sample call: `python scripts/visualization/visualize_metrics_pretraining.py`

#### visualize_datasets_mf1_testdata_cv.py

Creates the plot used in Figure 2 of the paper by reading the results of experiment groups exp001-exp006 from the `logs`
directory.

Sample call: `python scripts/visualization/visualize_nsubjects_vs_mf1_testdata_cv.py`

## Configuration

The configuration of an experiment is implemented using the [hydra](https://hydra.cc) framework that is based on YAML
files.
If you are not familiar with the hydra framework, you can find a good introduction and tutorial in the [official
documentation](https://hydra.cc/docs/intro).
This repository makes use of the object instantiation feature of hydra, which allows to instantiate objects at runtime
based on the configuration files (see [here](https://hydra.cc/docs/advanced/instantiate_objects/overview/) for more
details).

All configuration files must be placed within the `config` directory.
The configuration files are organized in a hierarchical structure, where the base configuration is defined in
`config/base_config.yaml`, the experiment-specific configurations are defined in the experiment folders (e.g.
`config/exp001/fpt_config.yaml` for exp001), and the
sub-experiment-specific configurations are defined in the folders of the sub-experiments (e.g.
`config/exp001/exp001a.yaml` for exp001a).
Configuration files that are lower in the hierarchy can overwrite parameters defined in higher-level configuration
files.
All configurations can be overwritten using command line arguments when running a script.

Any parameters or groups of parameters that should be `None`, have to be configured as either `null` or `Null` following
the YAML definition.

The available parameters are described in the existing configuration files and the doc-strings of the classes.
To get an overview of the final configuration of a run, it might be helpful to look into the `.hydra` folder in
the `logs` directory after running a script.
