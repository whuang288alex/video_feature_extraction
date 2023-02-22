## TODOs before submission

- To pack the environments:

```sh
conda activate base
conda pack -n feature_extraction
chmod 644 feature_extraction.tar.gz
```

- To pack your codes (Please change i3d_arch to the model that you are using)


#### (NOTE: For the code to run on CHTC clusters, you will need to download the pretrained checkpoints and include them in the assets folder. For more instruction, please refer to [models/README](models/README.md))



```sh
tar -zcvf code.tar.gz ../slurm.py ../config.py ../dataset.py ../extract_features.py ../configs/ ../models/*.py ../models/i3d_arch
```

- To move those to the staging directory:

```
mv feature_extraction.tar.gz code.tar.gz /staging/groups/li_group_biostats`
```

## Useful Condor commands

- To submit files: `condor_submit run_extraction.sub`

- To check out task status: `conqor_q`

- To check out the reason for holding: `condor_q -af HoldReason`

- To remove task:  `condor_rm id`

- To remove all tasks: `condor_rm $USER`
