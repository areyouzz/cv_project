"""
Compute global scale to align DA3 metric depth (meters) to COLMAP units.

For DA3-nested metric depth, we only need a single global scale factor,
not per-image scale/offset like with monocular relative depth.
"""

import numpy as np
import argparse
import cv2
import json
import os
from read_write_model import read_model, qvec2rotmat


def compute_global_scale(base_dir: str, depths_dir: str, max_depth_norm: float, model_type: str = "bin"):
    """
    Compute single global scale: COLMAP_depth = DA3_depth * scale
    
    Args:
        base_dir: Path to COLMAP dataset
        depths_dir: Path to DA3 depth maps
        max_depth_norm: The max_depth used for normalization in run.py
        model_type: COLMAP model format (bin or txt)
    
    Returns:
        global_scale: Scale factor to convert meters to COLMAP units
    """
    # Read COLMAP data
    cam_intrinsics, images_metas, points3d = read_model(
        os.path.join(base_dir, "sparse", "0"), ext=f".{model_type}"
    )
    
    # Build point cloud lookup
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs
    
    all_colmap_depths = []
    all_da3_depths = []
    
    for key in images_metas:
        image_meta = images_metas[key]
        cam = cam_intrinsics[image_meta.camera_id]
        
        # Get 3D point indices visible in this image
        pts_idx = image_meta.point3D_ids
        mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
        pts_idx = pts_idx[mask]
        valid_xys = image_meta.xys[mask]
        
        if len(pts_idx) < 10:
            continue
        
        # Get COLMAP depth (z in camera space)
        pts = points3d_ordered[pts_idx]
        R = qvec2rotmat(image_meta.qvec)
        pts_cam = np.dot(pts, R.T) + image_meta.tvec
        colmap_depth = pts_cam[:, 2]  # z-depth in COLMAP units
        
        # Load DA3 depth map
        n_remove = len(image_meta.name.split('.')[-1]) + 1
        depth_path = os.path.join(depths_dir, f"{image_meta.name[:-n_remove]}.png")
        
        if not os.path.exists(depth_path):
            continue
        
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            continue
        
        # Denormalize: 16-bit PNG -> meters
        depth_map = depth_map.astype(np.float32) / 65535.0 * max_depth_norm
        
        # Scale pixel coordinates to depth map resolution
        scale = depth_map.shape[0] / cam.height
        pixel_coords = (valid_xys * scale).astype(np.int32)
        
        # Filter valid coordinates
        valid = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < depth_map.shape[1]) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < depth_map.shape[0]) &
            (colmap_depth > 0.01)
        )
        
        if valid.sum() < 5:
            continue
        
        # Sample DA3 depth at sparse point locations
        px = pixel_coords[valid, 0]
        py = pixel_coords[valid, 1]
        da3_depth = depth_map[py, px]
        
        # Filter out invalid DA3 depths
        depth_valid = da3_depth > 0.01
        all_colmap_depths.extend(colmap_depth[valid][depth_valid])
        all_da3_depths.extend(da3_depth[depth_valid])
    
    all_colmap_depths = np.array(all_colmap_depths)
    all_da3_depths = np.array(all_da3_depths)
    
    if len(all_colmap_depths) < 10:
        print("Warning: Not enough correspondences, using scale=1.0")
        return 1.0
    
    # Compute robust global scale using median ratio
    # colmap_depth = da3_depth * scale
    ratios = all_colmap_depths / all_da3_depths
    global_scale = np.median(ratios)
    
    # Print statistics
    print(f"Correspondences: {len(all_colmap_depths)}")
    print(f"Global scale (COLMAP/meters): {global_scale:.4f}")
    print(f"Scale std: {np.std(ratios):.4f}")
    
    return global_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='Path to COLMAP dataset')
    parser.add_argument('--depths_dir', required=True, help='Path to DA3 depth maps')
    parser.add_argument('--max_depth', type=float, default=2.0, 
                        help='Max depth used in run.py normalization')
    parser.add_argument('--model_type', default='bin', help='COLMAP model type')
    args = parser.parse_args()
    
    global_scale = compute_global_scale(
        args.base_dir, args.depths_dir, args.max_depth, args.model_type
    )
    
    # Save simple scale file
    output = {
        "global_scale": float(global_scale),
        "description": "depth_colmap = depth_meters * global_scale"
    }
    
    output_path = os.path.join(args.base_dir, "sparse", "0", "depth_scale.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()