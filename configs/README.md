# Modifying/Adding config files

<b>Note: </b> here, a video is segmented into multiple "clips", and a feature will be extraced from each "clip".

- video_dir_path: The directory where the input videos are stored (all the videos in this directory will be processed)

- out_path: The directory where the extraced features will be stored

- batch_size: The number of "clip" that get fed into the model at each iteration

- frame_window: The number of frames in a "clip" 

- stride: The number of frames between the first frame of two neighboring "clips"

- use_remote: whether you want to use a remote checkpoints or a local one

- hub_path: the name of the checkpoints (can be either local or remote depending on "use_remote")

- side_size: This is the size that the shorter side of the input video will be resized to

- crop: For most models, you can choose between "three_crops" and "center". 
(To use more crops, please go to models/model_name.py and modify the get_transform method)

- crop_size: the size of the crop mentioned above.

- mean, std: the mean and standard deviation used for standardizing input

- model_module_str: the location of the python file where you implement the model (typically under `models/`)

# For changing num_worker and batch_size

Finding the best num_worker and batch_size can be kind of tricky, as using num_worker and batch_size that 
are too big can lead to some unexpected behavior from pytorch dataloader.