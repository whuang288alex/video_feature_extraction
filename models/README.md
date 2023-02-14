# Using pretrained models

- If you want to use the local weights (either pretrained or fine-tuned), set the variable 'use_remote' in config files to false.
- Models that REQUIRE you to manually download weights includes: egovlp and i3d. Please download the weights and add them to "models/model_name_arch/assets'
  - https://drive.google.com/drive/folders/1Qe_9XLUJELB69gYwSpAo1DU3yZ64BOpg
- Set the variable "hub_path" or "pretrained_dataset" to your desired pretrained model name.

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
