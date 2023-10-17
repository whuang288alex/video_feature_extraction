# Extracting Features from Videos

This directory contains the code to extract features from video datasets using mainstream vision models such as Slowfast, i3d, c3d, CLIP, etc. The only requirement for you is to provide a list of videos that you would like to extract features from in your input directory.

<br>

## Requirements

To set up the environment with conda, use the following commands:
```sh
conda create --name feature_extraction python=3.9
conda activate feature_extraction
python -m pip install -r requirements.txt
```

<br>

## Structure of the Repository

- ` models/`: This is the directory where you customize the models (e.g. structure, preprocessing, cropping) you want to use for feature extraction.

- ` configs/`: This directory contains configuration files that define different runtime settings
    
- ` videos/`: This is the default directory where you store the input videos.
    
- ` condor/`: This directory contains the code to submit computation to CHTC for throughput computing, ignore if not applicable.

<br>

## Feature Extraction 

(Make sure to change the path in the config file)

```sh
python main.py --config-name i3d_rgb_kinetics
```

<br>

##  To run with a different configuration

- Change `--config-name` to the name of the desired configuration file (under `configs/`) 

- Examples in main.sh

##  To use custom transformation, cropping, and mirroring

- Change the implementation of the get_transform method under models/[model_name].py

<br>

## Adding a Model

1. Add a new yaml file to `configs/`. Check out [configs/README](configs/README.md) for more instructions.

2. Add a new python file to `models/`. Check out [models/README](models/README.md) for more instructions.

<br>

## Important Note

For egovlp, i3d, and c3d, you are REQUIRED to manually download pretrained checkpoints for the code to start working. Please refer to [models/README](models/README.md) for more info.

<br>

## Acknowledgement

This directory is built on top of the <a href = "https://github.com/facebookresearch/Ego4d"> Ego4d directory </a>.

<br>

## Appendix

To resize the video or to convert the video into a different frame rate, refer to this <a href = "https://github.com/whuang288alex/resize_videos"> repository </a>.

