# Adding a model

- Create a python file with model name as file name, and include the following methods
  
  - ModelConfig(BaseModelConfig)
  
  - get_transform(config: ModelConfig)
   
  - load_model(config: ModelConfig)
  
  - addition functions for your model

- Useful functions in common.py

  - FeedVideoInput
  
  - FeedVideoInputList
  
  - Mirror
  
  - ThreeCrop
 
# Using Local Checkpoints

If you want to use the local checkpoints (either pretrained or fine-tuned)

1. Set the variable `use_remote` in config files to false.

2. Download the <a href = "https://drive.google.com/drive/folders/1Qe_9XLUJELB69gYwSpAo1DU3yZ64BOpg"> checkpoints </a>and add them to `./models/model_name_arch/assets`

3. Set the variable `hub_path` or `pretrained_dataset` in the config file to your desired pretrained checkpoints name.
