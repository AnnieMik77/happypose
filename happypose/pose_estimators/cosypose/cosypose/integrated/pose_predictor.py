from collections import defaultdict
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.pose_estimators.cosypose.cosypose.lib3d.cosypose_ops import (
    TCO_init_from_boxes,
    TCO_init_from_boxes_zup_autodepth,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger
from happypose.pose_estimators.cosypose.cosypose.utils.timer import Timer
from happypose.toolbox.inference.pose_estimator import PoseEstimationModule
from happypose.toolbox.inference.types import PoseEstimatesType

logger = get_logger(__name__)


class CoarseRefinePosePredictor(PoseEstimationModule):
    # TODO: deprecate in favor of PosePredictor
    def __init__(self, coarse_model=None, refiner_model=None, bsz_objects=64):
        super().__init__()
        self.coarse_model = coarse_model
        self.refiner_model = refiner_model
        self.bsz_objects = bsz_objects
        self.eval()

    @torch.no_grad()
    def batched_model_predictions(self, model, images, K, obj_data, n_iterations=1):
        timer = Timer()
        timer.start()

        ids = torch.arange(len(obj_data))

        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)

        preds = defaultdict(list)
        for (batch_ids,) in dl:
            timer.resume()
            obj_inputs = obj_data[batch_ids.numpy()]
            labels = obj_inputs.infos["label"].values
            im_ids = obj_inputs.infos["batch_im_id"].values
            images_ = images[im_ids]
            K_ = K[im_ids]
            TCO_input = obj_inputs.poses
            outputs = model(
                images=images_,
                K=K_,
                TCO=TCO_input,
                n_iterations=n_iterations,
                labels=labels,
            )
            timer.pause()
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs[f"iteration={n}"]

                infos = obj_inputs.infos
                batch_preds = tc.PandasTensorCollection(
                    infos,
                    poses=iter_outputs.TCO_output,
                    poses_input=iter_outputs.TCO_input,
                    K_crop=iter_outputs.K_crop,
                    boxes_rend=iter_outputs.boxes_rend,
                    boxes_crop=iter_outputs.boxes_crop,
                )
                preds[f"iteration={n}"].append(batch_preds)

        logger.debug(
            f"Pose prediction on {len(obj_data)} detections "
            f"(n_iterations={n_iterations}): {timer.stop()}",
        )
        preds = dict(preds)
        for k, v in preds.items():
            preds[k] = tc.concatenate(v)
        return preds

    def make_TCO_init(self, detections, K):
        K = K[detections.infos["batch_im_id"].values]
        boxes = detections.bboxes
        if self.coarse_model.cfg.init_method == "z-up+auto-depth":
            meshes = self.coarse_model.mesh_db.select(detections.infos["label"])
            points_3d = meshes.sample_points(2000, deterministic=True)
            TCO_init = TCO_init_from_boxes_zup_autodepth(boxes, points_3d, K)
        else:
            TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=boxes, K=K)
        return tc.PandasTensorCollection(infos=detections.infos, poses=TCO_init)

    def run_inference_pipeline(
        self,
        images,
        K,
        detections=None,
        data_TCO_init=None,
        n_coarse_iterations=1,
        n_refiner_iterations=1,
    ):
        preds = {}
        if data_TCO_init is None:
            assert detections is not None
            assert self.coarse_model is not None
            assert n_coarse_iterations > 0
            data_TCO_init = self.make_TCO_init(detections, K)
            coarse_preds = self.forward_coarse_model(
                images,
                K,
                data_TCO_init,
                n_coarse_iterations=n_coarse_iterations,
            )
            for n in range(1, n_coarse_iterations + 1):
                preds[f"coarse/iteration={n}"] = coarse_preds[f"iteration={n}"]
            data_TCO = coarse_preds[f"iteration={n_coarse_iterations}"]
        else:
            assert n_coarse_iterations == 0
            data_TCO = data_TCO_init
            preds["external_coarse"] = data_TCO

        if n_refiner_iterations >= 1:
            assert self.refiner_model is not None
            refiner_preds = self.forward_refiner(
                images,
                K,
                data_TCO,
                n_refiner_iterations=n_refiner_iterations,
            )
            for n in range(1, n_refiner_iterations + 1):
                preds[f"refiner/iteration={n}"] = refiner_preds[f"iteration={n}"]
            data_TCO = refiner_preds[f"iteration={n_refiner_iterations}"]
        return data_TCO, preds

    def forward_coarse_model(
        self,
        images,
        K,
        data_TCO_init,
        n_coarse_iterations,
    ) -> Tuple[PoseEstimatesType, dict]:
        return self.batched_model_predictions(
            self.coarse_model,
            images,
            K,
            data_TCO_init,
            n_iterations=n_coarse_iterations,
        )

    def forward_refiner(
        self,
        images,
        K,
        data_TCO,
        n_refiner_iterations,
    ) -> Tuple[dict, dict]:
        return self.batched_model_predictions(
            self.refiner_model,
            images,
            K,
            data_TCO,
            n_iterations=n_refiner_iterations,
        )
