- To submit files: `condor_submit run_extraction.sub`
- To check out task status: `conqor_q`
- To check out the reson for holding: `condor_q -af HoldReason`
- To remove task:  `condor_rm id`
- To remove all tasks: `condor_rm $USER`
- To pack your codes: `tar -zcvf code.tar.gz slurm.py config.py dataset.py extract_features.py configs/ features/ inputs models/`

