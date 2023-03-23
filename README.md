# Fast and Easy to use video feature extractor

This directory contains the code to extract features from video datasets using mainstream vision models uch as Slowfast, i3d, c3d, CLIP, etc. The pipeline is super simple as you only need to provide the path to the input directory that contains your video data.


## Requirements

To set up the environment with conda, use the following commands:
```sh
conda create --name feature_extraction python=3.9
conda activate feature_extraction
python -m pip install -r requirements.txt
```
## Structure of the Repository

- ` configs/`: This directory contains configuration files that define different runtime settings
    
- ` models/`: This is the directory where you implement the models you use for feature extraction.
    
- ` inputs/`: This is the default directory where you store the input videos.
    
- ` submit/`: This directory contains the code to submit computation to CHTC for throughput computing.

## Feature Extraction 

```sh
python slurm.py --config-name slowfast_r101_8x8
```

###  To run with a different configuration

Change `--config-name` to the name of the desired configuration file (under `configs/`) 

#### For example, to use the I3D model:

```bash
python slurm.py --config-name i3d_rgb_kinetics
```

#### To use the CLIP model:

```bash
python slurm.py --config-name clip_vit_b_32
```

#### (NOTE: for egovlp, i3d, and c3d, you are REQUIRED to manually download pretrained checkpoints for the code to start working. Please refer to [models/README](models/README.md) for more info.)


## Adding a Model

1. Add a new yaml file to `configs/`. Check out [configs/README](configs/README.md) for more instructions.

2. Add a new python file to `models/`. Check out [models/README](models/README.md) for more instructions.

## Acknowledgement

This directory is built on top of the <a href = "https://github.com/facebookresearch/Ego4d"> Ego4d directory </a>.

## Appendix

To resize the video:

`python resize_videos.py -vi ./inputs/videos -vo ./ -s 288 -fps 30`

