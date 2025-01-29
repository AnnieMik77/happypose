import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.pose_estimators.cosypose.cosypose.datasets.samplers import (
    DistributedSceneSampler,
)
from happypose.pose_estimators.cosypose.cosypose.utils.distributed import (
    get_rank,
    get_tmp_dir,
    get_world_size,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger

logger = get_logger(__name__)
# logger = get_file_logger(__name__, "multiview_runner.log")


class MultiviewRunner:
    def __init__(self, scene_ds, batch_size=1, cache_data=False, n_workers=4):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        assert batch_size == 1
        sampler = DistributedSceneSampler(
            scene_ds,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        self.sampler = sampler
        dataloader = DataLoader(
            scene_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

    def collate_fn(self, batch):
        batch_im_id = -1
        cam_infos, K, TWC = [], [], []
        im_infos = []
        det_infos, bboxes = [], []

        for n, data in enumerate(batch):
            assert n == 0
            images, masks, obss = data
            for _c, obs in enumerate(obss):
                batch_im_id += 1
                frame_info = obs["frame_info"]
                im_info = {
                    k: getattr(frame_info, k) for k in ("scene_id", "view_id", "group_id")
                }
                im_info.update(batch_im_id=batch_im_id)
                cam_info = im_info.copy()

                im_infos.append(im_info)

                K.append(obs["camera"].K)
                TWC.append(obs["camera"].TWC.matrix)
                cam_infos.append(cam_info)

                for _o, obj in enumerate(obs["objects"]):
                    obj_info = {
                        "label": obj.label,
                        "score": 1.0,
                    }
                    obj_info.update(im_info)
                    bboxes.append(obj.bbox_modal)
                    det_infos.append(obj_info)

        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
            TWC=torch.as_tensor(np.stack(TWC)),
        )
        data = {
            "images": images,
            "cameras": cameras,
            "im_infos": im_infos,
            "gt_detections": None, # remove this!!!
        }
        return data

    def get_predictions(
        self,
        pose_predictions=None,
        mv_predictor=None,
        use_known_camera_poses=False,
    ):
        predictions = defaultdict(list)
        for n, data in enumerate(tqdm(self.dataloader)):
            images = data["images"].cuda().float().permute(0, 3, 1, 2) / 255
            cameras = data["cameras"].cuda().float()
            im_infos = data["im_infos"]

            # logger.debug(f"{'-'*80}")
            # logger.debug(f"Predictions on {data['im_infos']}")

            def get_preds():
                torch.cuda.synchronize()
                # get only the detections for this batch
                
                batch_predictions = pose_predictions
                # TODO: filter by scene_id and view_id
                # Get scene_id and view_id from current batch
                batch_scene_ids = [info['scene_id'] for info in im_infos]
                batch_view_ids = [info['view_id'] for info in im_infos]

                # Create mask for matching scene_id and view_id
                scene_mask = batch_predictions.infos['scene_id'].isin(batch_scene_ids)
                view_mask = batch_predictions.infos['view_id'].isin(batch_view_ids)
                batch_mask = scene_mask & view_mask

                # Get indices where mask is True
                batch_indices = batch_mask.values.nonzero()[0]
                
                # Filter predictions using indices
                batch_predictions = tc.PandasTensorCollection(
                    infos=batch_predictions.infos.iloc[batch_indices].reset_index(drop=True),
                    **{k: v[batch_indices] for k,v in batch_predictions.tensors.items()}
                )

                # 1 for all detections
                batch_predictions.infos["group_id"] = im_infos[0]["group_id"]

                # put batch_predictions on GPU
                batch_predictions = batch_predictions.cuda()
                mv_preds = mv_predictor.predict_scene_state(
                    batch_predictions,
                    cameras,
                    use_known_camera_poses=use_known_camera_poses,
                )

                all_preds = {}
                all_preds["multiview"] = mv_preds["ba_output+all_cand"]
                return all_preds

            # Run once without measuring timing
            if n == 0:
                get_preds()
            all_preds = get_preds()

            # NOTE: time isn't correct for n iterations < max number of iterations
            for k, v in all_preds.items():
                v.infos = v.infos.loc[:, ["scene_id", "view_id",  "label", "score"]]
                predictions[k].append(v.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions
