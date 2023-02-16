# Using pretrained models

### Models that REQUIRE you to manually download weights includes: egovlp, i3d, and c3d

- If you want to use the local weights (either pretrained or fine-tuned), set the variable 'use_remote' in config files to false.

- Please download the <a href = "https://drive.google.com/drive/folders/1Qe_9XLUJELB69gYwSpAo1DU3yZ64BOpg"> weights </a>and add them to "./models/model_name_arch/assets'

- Set the variable "hub_path" and "pretrained_dataset" in the config file to your desired pretrained model name.

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
