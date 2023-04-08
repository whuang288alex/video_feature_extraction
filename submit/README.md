#### (NOTE: For the code to run on CHTC clusters, you will need to download the pretrained checkpoints and include them in the assets folder. For more instruction, please refer to [models/README](models/README.md))

#### (NOTE: You might need to change the paths in the configuration file to match the assumptions in run_extraction.sh)


## TODOs before submission

1. To pack the environments:

```sh
conda activate base
conda pack -n feature_extraction
chmod 644 feature_extraction.tar.gz
```

2. To pack your codes (Please change i3d_arch to the model that you are using)

```sh
tar -zcvf code.tar.gz ../slurm.py ../config.py ../dataset.py ../extract_features.py ../get_videos.py ../configs/ ../models/*.py ../models/i3d_arch
```

3. To move those to the staging directory:

```
mv feature_extraction.tar.gz  /staging/groups/li_group_biostats
mv code.tar.gz /staging/groups/li_group_biostats
```

4. To specify the configuration

change the config file name in `config_name.txt` to your desired config file name.

5. Submit the files

`condor_submit run_extraction.sub`

## Other Useful Condor commands

- To submit interactive job for debugging: `condor_submit -i run_extraction.sub`

- To check out task status: `conqor_q`

- To check out the reason for holding: `condor_q -af HoldReason`

- To remove task:  `condor_rm id`

- To remove all tasks: `condor_rm $USER`
