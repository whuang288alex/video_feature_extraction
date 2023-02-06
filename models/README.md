# Adding a model

- common.py
  - FeedVideoInput
  - FeedVideoInputList
  - Mirror
  - ThreeCrop
- Create a python file with model name as file name
  - ModelConfig, which must inherit from ego4d.features.model.base_model_config.BaseModelConfig
  - Additional configuration for your model
  - get_transform(config: ModelConfig)
  - load_model(config: ModelConfig)
