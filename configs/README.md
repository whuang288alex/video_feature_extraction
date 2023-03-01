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

- crop:
 
  - For most models, you can choose between "three_crops" and "center". 
  
  - For c3d, you can choose between "center", "five_crops", and "ten_crops".
  
  - For slowfast, there is no need to crop.

- crop_size: the size of the crop mentioned above.

- mean, std: the mean and standard deviation used for standardizing input

- model_module_str: the location of the python file where you implement the model (typically under `models/`)

# For changing num_worker and batch_size

Finding the best num_worker and batch_size can be kind of tricky, as changing num_worker and batch_size can lead to some unexpected behavior from pytorch dataloader (the exact number might varies across machines, below is the result on the shared GPU when no one else is using it). 

- clip	
  - only works if batch_size <= 32
  - only works if num_worker <= 8 for batch_size = 32

- slowfast
  - only works if batch_size <= 64
  - only works if num_worker <= 4 for batch_size = 64
 
- i3d
  - only works if batch_size <= 32
  - only works if num_worker <= 2 for batch_size = 32

- c3d
  - only works if batch_size <= 32
  - only works if num_worker <= 8 for batch_size = 32

- mvit
  - only works if batch_size <= 32
  - only works if num_worker <= 8 for batch_size = 32

- omnivore
  - only works if batch_size <= 4
  - only works if num_worker <= 8 for batch_size = 4

- egovlp 
  - only works if batch_size <= 128
  - only works if num_worker <= 8 for batch_size = 128
