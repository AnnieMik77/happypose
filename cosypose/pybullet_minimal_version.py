
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

########################
# Add cosypose to my path -> dirty
import sys
sys.path.insert(0, '/home/emaitre/cosypose')
########################

import cosypose

from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera

from cosypose_wrapper import CosyPoseWrapper


dataset_to_use = 'ycbv'  # tless or ycbv

IMG_RES = 640, 480 
# Realsense 453i intrinsics (from rostopic camera_info)
K_rs = np.array([615.1529541015625, 0.0, 324.5750732421875, 
    0.0, 615.2452392578125, 237.81765747070312, 
    0.0, 0.0, 1.0]).reshape((3,3))

img_dir = 'imgs'
# image_name = 'all_Color.png'
# image_name = 'all_far_Color.png'
# image_name = 'banana_Color.png'
image_name = 'cheezit_Color.png'
# image_name = 'wood_block_Color.png'

# imread stores color dim in the BGR order by default
brg = cv2.imread(img_dir + '/' + image_name)
# CosyPose uses a RGB representation internally?
rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)

cosy_pose = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)

import time
t = time.time()
preds = cosy_pose.inference(rgb, K_rs)
print('\nInference time (sec): ', time.time() - t)
print('Number of detections: ', len(preds))

print(type)
print("preds = ", preds)
print("poses =", preds.poses)
print("poses_input =", preds.poses_input)
print("k_crop =", preds.K_crop )
print("boxes_rend =", preds.boxes_rend)
print("boxes_crop =", preds.boxes_crop)


# rendering
renderer = BulletSceneRenderer('ycbv', gpu_renderer=False)
cam = {
    'resolution': IMG_RES,
    'K': K_rs,
    'TWC': np.eye(4),
}

# render_prediction_wrt_camera calls BulletSceneRenderer.render_scene using only one camera at pose Identity and return only rgb values
# BulletSceneRenderer.render_scene: gets a "object list" (prediction like object), a list of camera infos (with Km pose, res) and renders
# a "camera observation" for each camera/viewpoint
# Actually, renders: rgb, mask, depth, near, far
rgb_render = render_prediction_wrt_camera(renderer, preds, cam)
mask = ~(rgb_render.sum(axis=-1) == 0)

alpha = 0.1

rgb_n_render = rgb.copy()
rgb_n_render[mask] = rgb_render[mask]

# make the image background a bit fairer than the render
rgb_overlay = np.zeros_like(rgb_render)
rgb_overlay[~mask] = rgb[~mask] * 0.6 + 255 * 0.4
rgb_overlay[mask] = rgb_render[mask] * 0.8 + 255 * 0.2


cv2.imshow('raw img', brg)
# Detected object
#cv2.imshow('rgb_n_render', cv2.cvtColor(rgb_n_render, cv2.COLOR_RGB2BGR))
# Blured background version
cv2.imshow('rgb_overlay',  cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)