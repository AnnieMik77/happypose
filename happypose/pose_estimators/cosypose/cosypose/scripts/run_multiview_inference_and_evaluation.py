""" Computes multiview predictions with the cosypose multiview algorithm

    based on singleview predictions stored in a csv file.

    Evaluates the results using the BOP toolkit.

"""

# Loading singlepose csvs 
# TODO: this import is here only to get correct version of some c++ library
from happypose.pose_estimators.cosypose.cosypose.multiview_refactor.model_loader import (
    load_models
)
import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from bop_toolkit_lib import inout  # noqa

# Standard Library
import os
from pathlib import Path
import pandas as pd

import numpy as np
import torch

# Third Party
from omegaconf import OmegaConf

# TODO: bop workaround, remove when env is fixed
import bop_toolkit_lib
bop_toolkit_path = Path(bop_toolkit_lib.__file__).parent
project_bop_path = Path("bop_toolkit_lib")
if not project_bop_path.exists():
    # Create symlink
    os.symlink(bop_toolkit_path, project_bop_path)

# Configs
from happypose.pose_estimators.megapose.bop_config import BOP_CONFIG
from happypose.pose_estimators.megapose.config import (
    DEBUG_RESULTS_DIR,
    RESULTS_DIR,
)

# Loading datasets and scenes
from happypose.toolbox.datasets.datasets_cfg import (
    make_object_dataset,
    make_scene_dataset,
)
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase

# Multiview inference
from happypose.pose_estimators.cosypose.cosypose.datasets.wrappers.multiview_wrapper import (
    MultiViewWrapper,
)
from happypose.pose_estimators.cosypose.cosypose.evaluation.pred_runner.multiview_only_predictions import MultiviewRunner
from happypose.pose_estimators.cosypose.cosypose.integrated.multiview_predictor import (
    MultiviewScenePredictor,
)

# MegaPose
from happypose.pose_estimators.megapose.evaluation.runner_utils import format_results
from happypose.pose_estimators.megapose.evaluation.bop import ( 
    convert_results_to_bop,
    _run_bop_evaluation
)
from happypose.pose_estimators.megapose.evaluation.eval_config import (
    BOPEvalConfig,
    EvalConfig,
    FullEvalConfig,
    HardwareConfig,
    MultiviewConfig
)
from happypose.pose_estimators.megapose.evaluation.evaluation import (
    generate_save_key,
    get_save_dir,
)

# logging
from happypose.toolbox.utils.logging import get_logger, set_logging_level

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_cfg_debug(cfg: EvalConfig) -> FullEvalConfig:
    cfg.batch_size = 1
    cfg.n_frames = cfg.n_views
    cfg.n_scenes = cfg.batch_size
    cfg.save_dir = DEBUG_RESULTS_DIR
    return cfg


def load_poses_csv(ds_name, path_to_csv = None):
    """
    Load the poses from a csv file and return them in a format used by the cosypose multiview algorithm.

    Args:
        ds_name (str): The name of the dataset.
        path_to_csv (str): The path to the csv file.
    
    Returns: 
        all_dets (tc.PandasTensorCollection): The poses as a TensorCollection.
        method (str): The method used to generate the poses.
    """
    assert path_to_csv is not None
    assert Path(path_to_csv).is_file(), f"The file {path_to_csv} doesn't exist"

    method = path_to_csv.split("/")[-1].split("_")[0]

    # Load the poses
    dets = inout.load_bop_results(path_to_csv)
    dets_df = pd.DataFrame(dets)
    R = torch.tensor(dets_df.pop("R"))  # shape (n, 3, 3)
    t = torch.tensor(dets_df.pop("t")) # shape (n, 3, 1)

    # Divide translation by 1000 to convert to meters
    t = t / 1000

    # Poses to 4x4 matrices
    poses = torch.zeros(R.shape[0], 4, 4)
    poses[:, :3, :3] = R
    poses[:, :3, 3:4] = t
    poses[:, 3, 3] = 1

    # Create the TensorCollection
    dets_df = dets_df.rename(columns={"im_id": "view_id"})
    dets_df["label"] = dets_df["obj_id"].apply(lambda x: f"{ds_name.split(".")[0]}-obj_{x:06d}")
    dets_df = dets_df.drop(columns=["obj_id"])
    all_dets = tc.PandasTensorCollection(
        dets_df,
        poses=poses,
    )

    return method, all_dets
 
def run_multiview_inference(args):
    logger.info(f"{'-'*80}")
    for k, v in dict(args).items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scene_ds = make_scene_dataset(
        args.ds_name,
        load_depth=False,
        n_frames=args.n_frames,
        n_scenes=args.n_scenes,
    )

    ds_name_short = args.ds_name.split(".")[0]

    # Load mesh_db:
    object_ds = make_object_dataset(ds_name_short)
    mesh_db = MeshDataBase.from_object_ds(object_ds)

    # Load singleview predictions from a file:
    method, pose_predictions = load_poses_csv(ds_name_short, path_to_csv=args.single_view_pred_path)

    # save key
    hash_of_run = np.random.randint(0, 1000000)
    args.result_id = method + f"_multiview_nviews={args.n_views}_{str(hash_of_run)}"
    args.save_dir = str(Path(args.save_dir) / args.result_id  /cfg.ds_name)

    # Create the multiview dataset
    scene_ds_multi = MultiViewWrapper(scene_ds, n_views=args.n_views)

    # Run the multiview inference
    pred_runner = MultiviewRunner(
        scene_ds_multi,
        batch_size=args.batch_size,
        cache_data=False,
    )
    mv_predictor = MultiviewScenePredictor(mesh_db)

    all_preds = pred_runner.get_predictions(
        pose_predictions=pose_predictions,
        mv_predictor=mv_predictor,
        use_known_camera_poses=args.use_known_camera_poses,
    )

    logger.info(f"Done with inference on ds={args.ds_name}")
    logger.info(f"Predictions: {all_preds.keys()}")


    # Gather predictions to cpu
    for k, v in all_preds.items():
        all_preds[k] = v.cpu()
    

    # Save the results
    assert args.save_dir is not None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Finished inference on {args.ds_name}, setting={args.result_id}")
    results = format_results(all_preds, {}, {})

    # Save the results all together:
    results_path = Path(save_dir)/ "results.pth.tar"
    torch.save(results, results_path)
    (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    scene_ds_multi.to_file(folder_name=save_dir)
    logger.info(f"Saved results in {save_dir}")

    return {
        "pred_keys": list(all_preds.keys()),
        "save_dir": save_dir,
        "results_path": results_path,
    }


def main(cfg: MultiviewConfig) -> None:
    bop_eval_cfgs = []

    # Create the correct config for dataset
    cfg.ds_name = BOP_CONFIG[cfg.ds_name]["inference_ds_name"][0]
    
    # Get the inference results
    if not cfg.skip_inference:
        # Run multiview inference, the results get saved to disk
        eval_out = run_multiview_inference(cfg)
    else:  
        # Otherwise load the previously inferenced output
        assert cfg.result_id is not None, "Set result_id to load the results"
        results_dir = Path(cfg.save_dir) / cfg.result_id / cfg.ds_name
        pred_keys = ["ba_output"] # TODO: load from predictions
        eval_out = {
            "pred_keys": pred_keys,
            "save_dir": results_dir,
            "results_path": results_dir / "results.pth.tar",
        }

        assert Path(
            eval_out["results_path"],
        ).is_file(), f"The file {eval_out['results_path']} doesn't exist"

    # Run the bop eval for some types of predictions
    if cfg.run_bop_eval:
        # eval only output of cosypose
        bop_eval_keys = {"ba_output"}
        bop_eval_keys = bop_eval_keys.intersection(set(eval_out["pred_keys"]))

        for method in bop_eval_keys:
            bop_eval_cfg = BOPEvalConfig(
                results_path=eval_out["results_path"],
                dataset=cfg.ds_name,
                split="test",
                eval_dir= Path(eval_out["save_dir"]) / "bop_evaluation",
                method=method,
                convert_only=False,
                use_post_score=False,
            )
            bop_eval_cfgs.append(bop_eval_cfg)

        # Run the bop eval
        for bop_eval_cfg in bop_eval_cfgs:
            results_path = bop_eval_cfg.results_path

            pose_method = cfg.result_id.replace("/", "-")
            pose_method = pose_method.replace("_", "-")
            dataset = bop_eval_cfg.dataset.split(".")[0]
            csv_path = bop_eval_cfg.eval_dir / f"{pose_method}-{bop_eval_cfg.method}_{dataset}-{bop_eval_cfg.split}.csv"
            
            convert_results_to_bop(results_path, csv_path, bop_eval_cfg.method, use_pose_score=False)
            _run_bop_evaluation(csv_path, bop_eval_cfg.eval_dir, eval_detection=False)           


if __name__ == "__main__":
    # TODO: logs are shown twice
    set_logging_level("debug")

    # Load config
    cli_cfg = OmegaConf.from_cli()
    logger.info(f"CLI config: \n {OmegaConf.to_yaml(cli_cfg)}")

    cfg: MultiviewConfig = OmegaConf.structured(MultiviewConfig)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Verify that the config is valid
    if not cfg.skip_inference:
        assert cfg.single_view_pred_path is not None
        assert cfg.n_views is not None
    else:
        assert cfg.result_id is not None

    assert cfg.ds_name is not None

    cfg.save_dir = RESULTS_DIR

    # TODO: work on debug mode
    if cfg.debug:
        cfg = update_cfg_debug(cfg)

    main(cfg)
