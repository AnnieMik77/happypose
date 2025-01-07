""" 
    this is here temporarily for loading models for cosypose.
    it has to be updated in all cosypose scripts due to new object dataset and scene dataset
"""

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import yaml

from happypose.pose_estimators.cosypose.cosypose.bop_config import (
    BOP_CONFIG,
    PBR_COARSE,
    PBR_DETECTORS,
    PBR_REFINER,
    SYNT_REAL_COARSE,
    SYNT_REAL_DETECTORS,
    SYNT_REAL_REFINER,
)
from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR, RESULTS_DIR
from happypose.toolbox.datasets.datasets_cfg import (
    make_object_dataset,
    make_scene_dataset,
)
from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector

from happypose.pose_estimators.cosypose.cosypose.integrated.pose_predictor import (
    CoarseRefinePosePredictor,
)

# Pose estimator
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase


# Detection
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    create_model_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    load_model_cosypose,
)

from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger
from happypose.toolbox.renderer.bullet_batch_renderer import BulletBatchRenderer

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def load_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set="tless"):
    # TODO: This should not be hardcoded here, it limits the use to tless and ycbv
    # also, the other values are not used...
    if object_set == "tless.bop19":
        ds_name_short, object_ds_name, urdf_ds_name = "tless", "tless.bop19", "tless.cad"
    else:
        ds_name_short, object_ds_name, urdf_ds_name = "ycbv", "ycbv.bop-compat.eval", "ycbv"


    object_ds = make_object_dataset(ds_name_short)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_ds, n_workers=n_workers, preload_cache=False)
    mesh_db_batched = mesh_db.batched().cuda()

    # TODO: verify this checking of confing file
    # run_dir = EXP_DIR / coarse_run_id
    # cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader)
    # cfg = check_update_config_pose(cfg)



    # TODO: load_model_cosypose is also from CosyPose and not HappyPose
    if coarse_run_id is None:
        coarse_model = None
    else:
        coarse_model = load_model_cosypose(
        EXP_DIR / coarse_run_id, renderer, mesh_db_batched, device
    )
    refiner_model = load_model_cosypose(
        EXP_DIR / refiner_run_id, renderer, mesh_db_batched, device
    )
    # TODO: this is only think from CosyPose which is not updated
    model = CoarseRefinePosePredictor(
        coarse_model=coarse_model,
        refiner_model=refiner_model,
    )
    return model, mesh_db
