# Video Features

This directory contains code to extract features from video datasets using different models.

## Requirements

To install conda on your remote Linux server, use the following commands:
```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

To set up the environment with conda, use the following commands:
```sh
conda create --name feature_extraction python=3.9
conda activate feature_extraction
python -m pip install -r requirements.txt
```
## Directory setup

- ` configs/`: this is the directory where you store configuration files that define different runtime settings
- ` models/`: this is the directory where you implement the models you use for feature extraction.
- ` inputs/`: this is the default directory where you store the input videos.
- ` features/`: this is the default directory where you store the features you extract.

## Feature Extraction

```sh
python slurm.py --config-name slowfast_r101_8x8
```

#### 1. To run with a different configuration

Provide `--config-name "name"`  where `"name"` is the name of the configuration file without the `.yaml` extension.

#### 2. To run on a subset of videos

Provide `io.uid_list` in the YAML (`InputOutputConfig.uid_list`) or as a list of arguments on the CLI.

```bash
python slurm.py --config-name slowfast_r101_8x8 io.uid_list="[000a3525-6c98-4650-aaab-be7d2c7b9402]"
```

## Adding a Model

1. Add a new yaml file to `configs/`. Check out [configs/README](configs/README.md) for more instructions.
2. Add a new python file to `models/`. Check out [models/README](models/README.md) for more instructions.

## Acknowledgement

This directory is built on top of the <a href = "https://github.com/facebookresearch/Ego4d"> Ego4d<a> directory.
