## TODOs before submission

- To pack the environments:
    - `conda activate base`
    - `conda pack -n feature_extraction`
    - `chmod 644 feature_extraction.tar.gz`
- To pack your codes
    - `tar -zcvf code.tar.gz slurm.py config.py dataset.py extract_features.py configs/ features/ inputs models/`

- To move those to staging directory:
    - `mv feature_extraction.tar.gz code.tar.gz /staging/groups/li_group_biostats`

## Useful Condor commands

- To submit files: `condor_submit run_extraction.sub`
- To check out task status: `conqor_q`
- To check out the reson for holding: `condor_q -af HoldReason`
- To remove task:  `condor_rm id`
- To remove all tasks: `condor_rm $USER`
