#### NOTE: For the code to run on CHTC clusters, you will need to download the pretrained checkpoints and include them in the assets folder. For more instruction, please refer to [models/README](models/README.md)

## TODOs before submission

1. To pack the environments:

```sh
conda activate base
conda pack -n feature_extraction
chmod 644 feature_extraction.tar.gz
mv feature_extraction.tar.gz  /staging/groups/li_group_biostats/video_feature_extraction
```

2. To pack your codes (Please change clip_arch to the model that you are using)

```sh
tar -zcvf video_feature_extraction.tar.gz ../main.py ../src/ ../configs/ ../models/
mv video_feature_extraction.tar.gz /staging/groups/li_group_biostats/video_feature_extraction
```

3. To specify the configuration

Modify the bash script to run with the desired config.

5. Submit the files

`condor_submit extract.sub`

## Other Useful Condor commands

- To submit interactive job for debugging: `condor_submit -i extract.sub`

- To check out task status: `conqor_q`

- To check out the reason for holding: `condor_q -af HoldReason`

- To remove task:  `condor_rm id`

- To remove all tasks: `condor_rm $USER`
