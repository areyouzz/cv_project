#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
# import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # resize_shape = (resolution[1], resolution[0]) 

    # # mask
    # if cam_info.gt_alpha_mask is not None:
    #     import torch
    #     mask_np = np.array(cam_info.gt_alpha_mask)
    #     mask_tensor = torch.from_numpy(mask_np)

    #     # 做插值
    #     if mask_tensor.dim() == 2: # [H, W]
    #         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    #     elif mask_tensor.dim() == 3: # [H, W, C]
    #          mask_tensor = mask_tensor.permute(2,0,1)[:1, ...].unsqueeze(0)
    #     mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=resize_shape, mode='nearest')

    #     loaded_mask = mask_tensor.squeeze(0) # [1, H, W]

    # # depth
    # loaded_depth = None
    # if cam_info.gt_depth is not None:
    #     import torch
    #     depth_np = np.array(cam_info.gt_depth)
    #     depth_tensor = torch.from_numpy(depth_np)

    #     if depth_tensor.dim() == 2:
    #         depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    #     elif depth_tensor.dim() == 3:
    #         # 如果通道在最后 (H, W, 3)
    #         if depth_tensor.shape[2] <= 4: 
    #             depth_tensor = depth_tensor.permute(2, 0, 1) # -> [C, H, W]
            
    #         # 只取第一个通道，并增加 Batch 维 -> [1, 1, H, W]
    #         depth_tensor = depth_tensor[:1, :, :].unsqueeze(0)

    #     # 做插值
    #     depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=resize_shape, mode='bilinear', align_corners=False)

    #     loaded_depth = depth_tensor.squeeze(0)

    # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
    #               FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
    #               image=gt_image, gt_alpha_mask=loaded_mask,
    #               image_name=cam_info.image_name, uid=id, data_device=args.data_device, gt_depth=loaded_depth)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry