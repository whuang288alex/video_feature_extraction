# Adding a config

<b>Note: </b> here, a  video is divided into multiple "clips", and each "clip" is treated as a feature of the video

- fps: The number of frames per second in the input video
- video_dir_path: The path where the input video is stored
- out_path: The path where the extraced features will be stored
- batch_size: ?
- num_workers: ?
- frame_window: The number of frames in a clip 
- stride: The number of frames between the first frame of two neighboring clips
- run_locally: ?
- gpus_per_node: ?
- cpus_per_task: ?
- hub_path: ?
- side_size: ?
- dilation: ?
- mean & std: the mean and  standard deviation used for standardizing input, should be the same as the one used when traning the original model
- model_module_str: the python file location
