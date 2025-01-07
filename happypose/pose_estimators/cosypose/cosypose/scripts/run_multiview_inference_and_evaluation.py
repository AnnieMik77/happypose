""" Computes multiview predictions with the cosypose multiview algorithm

    based on singleview predictions stored in a csv file.

    Evaluates the results using the BOP toolkit.

"""

# Loading singlepose csvs
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
import torch.multiprocessing

# Third Party
from omegaconf import OmegaConf

# TODO: Bop workaround, remove when env is fixed
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

# TODO: this might be useful when using camera poses
from happypose.pose_estimators.cosypose.cosypose.lib3d.cosypose_ops import (
    TCO_init_from_boxes,
    TCO_init_from_boxes_zup_autodepth,
)

# MegaPose
from happypose.pose_estimators.megapose.config import (
    DEBUG_RESULTS_DIR,
    RESULTS_DIR,
)
from happypose.pose_estimators.megapose.evaluation.runner_utils import format_results
from happypose.pose_estimators.megapose.evaluation.bop import run_evaluation
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

# Distributed
from happypose.toolbox.utils.distributed import (
    get_rank,
    get_tmp_dir,
    get_world_size,
    init_distributed_mode,
)
from happypose.toolbox.utils.logging import get_logger, set_logging_level

# torch.multiprocessing.set_sharing_strategy("file_system")
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_cfg_debug(cfg: EvalConfig) -> FullEvalConfig:
    cfg.batch_size = 1
    cfg.n_frames = cfg.n_views
    cfg.n_scenes = cfg.batch_size * cfg.hardware.n_gpus

    assert cfg.result_id is not None
    cfg.save_dir = str(DEBUG_RESULTS_DIR / cfg.result_id)
    return cfg


def load_poses_csv(ds_name, path_to_csv = None):
    """
    Load the poses from a csv file and return them in a format used by the cosypose multiview algorithm.

    Args:
        ds_name (str): The name of the dataset.
        path_to_csv (str): The path to the csv file.
    
    Returns: 
        all_dets (tc.PandasTensorCollection): The poses as a TensorCollection.
    """
    assert path_to_csv is not None

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

    return all_dets
 
def run_multiview_inference(args):
    assert args.n_views > 2
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
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
    pose_predictions = load_poses_csv(ds_name_short, path_to_csv=args.single_view_pred_path)

    # save key
    save_key = cfg.single_view_method_name + "_multiview" + f"_nviews={args.n_views}"
    args.save_dir = str(Path(args.save_dir) / save_key)

    # Create the multiview dataset
    scene_ds_multi = MultiViewWrapper(scene_ds, n_views=args.n_views)

    # Run the multiview inference
    # TODO: solve parallelization without n_workers
    pred_runner = MultiviewRunner(
        scene_ds_multi,
        batch_size=args.batch_size,
        cache_data=False,
        n_workers=args.n_workers
    )
    mv_predictor = MultiviewScenePredictor(mesh_db)
    pred_kwargs = {
                    "pose_predictions": pose_predictions,
                    "mv_predictor": mv_predictor,
                }
    all_preds = pred_runner.get_predictions(**pred_kwargs)

    logger.info(f"Done with inference on ds={args.ds_name}")
    logger.info(f"Predictions: {all_preds.keys()}")


    # Gather predictions from different processes
    logger.info("Waiting on barrier.")
    torch.distributed.barrier()
    logger.info("Gathering predictions from all processes.")
    for k, v in all_preds.items():
        all_preds[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()
    
    torch.distributed.barrier()
    logger.info("Finished gathering predictions from all processes.")

    # Save the results
    if get_rank() == 0:
        results_path = args.save_dir / "results.pth.tar"
        assert args.save_dir is not None
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Finished inference on {args.ds_name}, setting={save_key}")
        results = format_results(all_preds, {}, {})
        torch.save(results, results_path)
        torch.save(results.get("summary"), save_dir / "summary.pth.tar")
        torch.save(results.get("predictions"), save_dir / "predictions.pth.tar")
        torch.save(results.get("dfs"), save_dir / "error_dfs.pth.tar")
        torch.save(results.get("metrics"), save_dir / "metrics.pth.tar")
        (save_dir / "summary.txt").write_text(results.get("summary_txt", ""))
        (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        logger.info(f"Saved predictions+metrics in {save_dir}")

        return {
            "results": results,
            "pred_keys": list(all_preds.keys()),
            "save_dir": save_dir,
            "results_path": results_path,
        }

    else:
        return None



def main(cfg: MultiviewConfig) -> None:
    bop_eval_cfgs = []

    init_distributed_mode()
    print("World size", get_world_size())

    # Create the correct config for dataset
    cfg.ds_name = BOP_CONFIG[cfg.ds_name]["inference_ds_name"][0]
    cfg.save_dir = Path(cfg.save_dir) / cfg.ds_name

    # Run multiview inference
    # Note that the results get saved to disk
    if not cfg.skip_inference:
        eval_out = run_multiview_inference(cfg)

    else:  
        # Otherwise load the previously inferenced output
        if get_rank() == 0:
            save_key = cfg.single_view_method_name + f"multiview_nviews={cfg.n_views}"

            results_dir = Path(cfg.save_dir) / save_key
            pred_keys = ["multiview"]
            eval_out = {
                "results_path": results_dir / "results.pth.tar",
                "pred_keys": pred_keys,
                "save_dir": results_dir,
            }

            assert Path(
                eval_out["results_path"],
            ).is_file(), f"The file {eval_out['results_path']} doesn't exist"

    # Run the bop eval for each type of prediction
    if cfg.run_bop_eval and get_rank() == 0:
        bop_eval_keys = {"multiview"}
        bop_eval_keys = bop_eval_keys.intersection(set(eval_out["pred_keys"]))

        for method in bop_eval_keys:
            if "bop19" not in cfg.ds_name:
                continue

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

    # Run the bop eval for each config
    if get_rank() == 0:
        if cfg.run_bop_eval:
            for bop_eval_cfg in bop_eval_cfgs:
                run_evaluation(bop_eval_cfg)

    logger.info(f"Process {get_rank()} reached end of script")

if __name__ == "__main__":
    print("Running eval")
    set_logging_level("debug")

    # Load config
    cli_cfg = OmegaConf.from_cli()
    logger.info(f"CLI config: \n {OmegaConf.to_yaml(cli_cfg)}")

    cfg: MultiviewConfig = OmegaConf.structured(MultiviewConfig)
    cfg.hardware = HardwareConfig(
        n_cpus=int(os.environ.get("N_CPUS", 10)),
        n_gpus=int(os.environ.get("WORLD_SIZE", 1)),
    )

    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Verify that the config is valid
    assert cfg.single_view_pred_path is not None
    assert cfg.single_view_method_name is not None
    assert cfg.result_id is not None
    assert cfg.ds_name is not None

    cfg.save_dir = RESULTS_DIR / cfg.result_id

    # TODO: really work on debug mode
    if cfg.debug:
        cfg = update_cfg_debug(cfg)

    print(cfg)
    main(cfg)
