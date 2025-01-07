# Multiview pose estimation

## Instal dependencies
TODO

## Run multiview for single view poses:

1. download the single view predictions or predict them
format should be csv
todo: specify their format

2. 
specify params of your experiment:
"single_view_pred_path=~/multiview_proj/mock_data_dir/results/ycbv-debug/ycbv.bop19/downloaded/foundpose_ycbv-test_733a8c68-39a4-4e6d-bb4d-8bfa8110ccba.csv",
"single_view_method_name=found_pose",
"ds_name=ycbv",
"result_id=multiview-debug",
"debug=False",

more params are in the config file


3. run the script:
/local2/homes/mikesann/multiview_proj/happypose_fork/happypose/happypose/pose_estimators/cosypose/cosypose/scripts/run_multiview_inference_and_evaluation.py


## TODO:
from happypose.pose_estimators.cosypose.cosypose.evaluation.pred_runner.multiview_only_predictions import MultiviewRunner
-> clean up this pred runner.
play with dependencies and fix them
multiprocessing and running on gpu maybe isn't working
add fixed camera poses
learn visualization of the resutls

