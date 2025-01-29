# Multiview pose estimation
Run cosypose multiview on a csv file containing estimated poses from BOP leaderboard.
Will run multiview algorithm and BOP evaluation. 

## Install dependencies
Should be the same as for cosypose

## Run multiview for single view poses:
0. `export HAPPYPOSE_DATA_DIR=/somewhere/convenient`
1. Download the single view predictions `csv` from BOP Leaderboard.

2. Parameters:\
`./happypose/pose_estimators/megapose/evaluation/eval_config/` contains list of optional params in `MultiviewConfig` class

3. run the script:
```
python ./happypose/pose_estimators/cosypose/cosypose/scripts/run_multiview_inference_and_evaluation.py \
single_view_pred_path=path/to/csv/method_ycbv-test_hash.csv  \
ds_name=ycbv \
debug=False \
use_known_camera_poses=True
```


## TODO:
- work on logger and debug mode
- work on workers/parallelization
- cean up pred runner: from happypose.pose_estimators.cosypose.cosypose.evaluation.pred_runner.multiview_only_predictions import MultiviewRunner
- track dependencies and fix them
- logging when ransac fails for known poses

