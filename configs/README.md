# Adding a config file

<b>Note: </b> here, a  video is divided into multiple "clips", and the result got from each "clip" is treated as a feature of the video

- fps: The number of frames per second in the input video

- video_dir_path: The path where the input video is stored

- out_path: The path where the extraced features will be stored

- batch_size: The number of clip that get fed into the model each time

- frame_window: The number of frames in a clip 

- stride: The number of frames between the first frame of two neighboring clips

- use_remote: determine if you are going to use a remote pretrained model or a local one

- hub_path: This is the url where you load the pretrained models (can be either local or remote depending on "use_remote")

- side_size: This is the size that the shorter side of the input video will be resized to

- crop:
 
  - For most models, you can choose between "three_crops" and "center". 
  
  - For c3d, you can choose between "center", "five_crops", and "ten_crops".
  
  - For slowfast, there is no need to crop.

- crop_size: the size of the crop mentioned above.

- dilation: number of frame to extract in each clip

- mean & std: the mean and  standard deviation used for standardizing input, should be the same as the one used when traning the original model

- model_module_str: the python file location

# Bugs to be fixed

Finding the best num_worker and batch_size can be kind of tricky, as changing num_worker and batch_size can lead to some unexpected behavior from pytorch dataloader due to machine performance issue (the exact number might varies across machines, below is the result on the shared GPU when no one else is using it). 

Rules of thumb: first set num_worker to zero and adjust batch_size to a number that doesn't cause "CUDA out of memory error".
Then, fix batch_size and adjust num_worker to a number that doesn't cause "Worker getting killed error". If setting num_worker is too cumbersome for you, just set it to zero as this should always work.

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
