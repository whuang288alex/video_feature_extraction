# Adding a model

- Ensure you have the following:
  - ModelConfig, which must inherit from ego4d.features.model.base_model_config.BaseModelConfig
  - Additional configuration for your model
  - get_transform(config: ModelConfig)
  - load_model(config: ModelConfig)
