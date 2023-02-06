# Adding a model

- Create a python file with model name as file name
  - ModelConfig(BaseModelConfig)
  - get_transform(config: ModelConfig)
  - load_model(config: ModelConfig)
  - addition functions for your model

- Useful functions in common.py
  - FeedVideoInput
  - FeedVideoInputList
  - Mirror
  - ThreeCrop
