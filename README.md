# Video Features

This directory contains code to extract features from video datasets using different models.

## Requirements

To set up the environment with conda, use the following commands:
```sh
conda create --name feature_extraction
conda activate feature_extraction
pip install -r requirements.txt
```

## Test Extraction

Running a test extraction to ensure you have everything setup right:

```sh
python3 profile.py --config-name slowfast_r101_8x8 schedule_config.run_locally=1
```

## Actual Extraction

```sh
python3 slurm.py --config-name slowfast_r101_8x8
```

#### 1. To run with a different configuration

Provide `--config-name "name"`  where `"name"` is the name of the configuration file without the `.yaml` extension.

#### 2. To run on a subset of videos

Provide `io.uid_list` in the YAML (`InputOutputConfig.uid_list`) or as a list of arguments on the CLI.

```bash
python3 slurm.py --config-name slowfast_r101_8x8 io.uid_list="[000a3525-6c98-4650-aaab-be7d2c7b9402]"
```

## Adding a Model

1. Add a new yaml file to `configs/`
2. Add a new python file to `models/`
3. Ensure you have the following:
    - ModelConfig, which must inherit from `model.base_model_config.BaseModelConfig`
        - Additional configuration for your model
    - get_transform(config: ModelConfig)
    - load_model(config: ModelConfig)

## Acknowledgement

This directory is built on top of the <a href = "https://github.com/facebookresearch/Ego4d"> Ego4d<a> directory.
