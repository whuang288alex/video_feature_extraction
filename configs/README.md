# Adding a config

- fps: 30
- video_dir_path: /inputs/videos
- out_path: /features/slowfast
- batch_size: 1
- num_workers: 8
- frame_window: 32
- stride: 8
- run_locally: true
- gpus_per_node: 1
- cpus_per_task: 10
- hub_path: slowfast_r101
- side_size: 288
- dilation: 2 
- mean:
- std:
- model_module_str: models.slowfast
