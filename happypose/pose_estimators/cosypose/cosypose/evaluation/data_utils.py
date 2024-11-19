from collections import defaultdict

import pandas as pd
import torch

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.toolbox.lib3d.transform_ops import invert_transform_matrices


def parse_obs_data(obs):
    data = defaultdict(list)
    frame_info = obs["frame_info"]
    TWC = torch.as_tensor(obs["camera"].TWC.matrix).float()
    for n, obj in enumerate(obs["objects"]):
        info = {
            "frame_obj_id": n,
            "label": obj.label,
            "visib_fract": obj.visib_fract,
            "scene_id": frame_info.scene_id,
            "view_id": frame_info.view_id,
        }
        data["infos"].append(info)
        data["TWO"].append(obj.TWO.matrix)
        data["bboxes"].append(obj.bbox_modal)

    for k, v in data.items():
        if k != "infos":
            data[k] = torch.stack([torch.as_tensor(x).float() for x in v])

    data["infos"] = pd.DataFrame(data["infos"])
    TCO = invert_transform_matrices(TWC).unsqueeze(0) @ data["TWO"]

    data = tc.PandasTensorCollection(
        infos=data["infos"],
        TCO=TCO,
        bboxes=data["bboxes"],
        poses=TCO,
    )
    return data


def data_to_pose_model_inputs(data):
    TXO = data.poses
    obj_infos = []
    for n in range(len(data)):
        obj_info = {"name": data.infos.loc[n, "label"]}
        obj_infos.append(obj_info)
    return TXO, obj_infos
