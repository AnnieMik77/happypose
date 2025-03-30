import numpy as np
import pandas as pd
import torch
import json

from .base import SceneDatasetWrapper


class MultiViewWrapper(SceneDatasetWrapper):  
    def __init__(self, scene_ds, n_views=4, mode="random"):
        if mode == "random":
            n_max_views = n_views
            frame_index = scene_ds.frame_index.copy().reset_index(drop=True)
            groups = frame_index.groupby(["scene_id"]).groups

            random_state = np.random.RandomState(0)
            self.frame_index = []
            for scene_id, group_ids in groups.items():
                n_max_views = n_views
                group_ids = random_state.permutation(group_ids)
                len_group = len(group_ids)
                for _k, m in enumerate(np.arange(len_group)[::n_max_views]):
                    ids_k = np.arange(len(group_ids))[m : m + n_max_views].tolist()
                    ds_ids = group_ids[ids_k]
                    df_group = frame_index.loc[ds_ids]
                    self.frame_index.append(
                        {
                            "scene_id": scene_id,
                            "view_ids": df_group["view_id"].values.tolist(),
                            "n_views": len(df_group),
                            "scene_ds_ids": ds_ids,
                        },
                    )

            self.frame_index = pd.DataFrame(self.frame_index)
            self.frame_index["group_id"] = np.arange(len(self.frame_index))
            self.scene_ds = scene_ds
            # self.init_from_file(scene_ds, "multiview_wrapper.json")
            self.to_file(f"ycbv_views_{n_views}.json")
        else:
            self.init_from_file(scene_ds, "ycbv_views_mapping_for_martin.json")


    def __getitem__(self, idx):
        row = self.frame_index.iloc[idx]
        ds_ids = row["scene_ds_ids"]
        rgbs, masks, obss = [], [], []
        for ds_id in ds_ids:
            rgb, mask, obs = self.scene_ds[ds_id]
            rgbs.append(rgb)
            masks.append(torch.stack(mask))
            obs["frame_info"].group_id = row["group_id"]
            obss.append(obs)

        rgbs = torch.tensor(rgbs)
        masks = torch.stack(masks)
        return rgbs, masks, obss

    def __len__(self):
        return len(self.frame_index)
    
    def init_from_file(self, scene_ds, file_path):
        self.scene_ds = scene_ds
        view_info = json.load(open(file_path))
        frame_index = pd.DataFrame(view_info)
        # add empty column for "scene_ds_ids"
        frame_index["scene_ds_ids"] = None

        for i, row in frame_index.iterrows():
            scene_id = row["scene_id"]
            view_ids = row["views"]
            scene_ds_ids = []
            for view_id in view_ids:
                # get row id
                ds_id = scene_ds.frame_index[
                    (scene_ds.frame_index["scene_id"] == scene_id)
                    & (scene_ds.frame_index["view_id"] == view_id)
                ].index[0]
                scene_ds_ids.append(ds_id)
            frame_index.at[i, "scene_ds_ids"] = scene_ds_ids

        # rename id column to group_id
        frame_index = frame_index.rename(columns={"id": "group_id"})
        frame_index = frame_index.rename(columns={"views": "view_ids"})


        self.frame_index = frame_index
        return self


    def to_file(self, file_path):
        view_info = self.frame_index.to_dict(orient="records")
        # for each "scene_ds_ids" remove this column
        for v in view_info:
            v.pop("scene_ds_ids")
        print("Saving view info to", file_path)
        json.dump(view_info, open(file_path, "w"))