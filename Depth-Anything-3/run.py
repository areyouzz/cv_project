import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

def get_image_list(img_path: str) -> list[str]:
    extentions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in extentions:
        files.extend(glob.glob(os.path.join(img_path, ext)))
        files.extend(glob.glob(os.path.join(img_path, '**', ext), recursive=True))
    return sorted(list(set(files)))

def save_depth(depth: np.ndarray, save_path: str, max_depth: float=2.5) -> float:
    depth_clipped = np.clip(depth, 0, max_depth)
    normalized = depth_clipped / max_depth

    # save as 16-bit PNG
    depth = (normalized * 65535).astype(np.uint16)
    cv2.imwrite(save_path, depth)

    return max_depth

def save_depth_metadata(output_dir: str, max_depth: float, image_names: list[str]):
    import json
    metadata = {
        "max_depth": max_depth,
        "depth_format": "metric_linear",
        "units": "meters",
        "images": image_names
    }
    with open(os.path.join(output_dir, "depth_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def process_batch(
    model: DepthAnything3,
    image_paths: list[str],
    output_dir: str,
    conf_output_dir: str | None,
    process_res: int,
    use_ray_pose: bool,
    max_depth: float,
) -> list[str]:
    
    prediction = model.inference(
        image=image_paths,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        use_ray_pose=use_ray_pose,
        ref_view_strategy="saddle_balanced",
        align_to_input_ext_scale=False,
    )

    saved_names = []
    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem
        depth_save_path = os.path.join(output_dir, f"{img_name}.png")

        depth = prediction.depth[i]
        save_depth(depth, depth_save_path, max_depth=max_depth)
        saved_names.append(img_name)

        if conf_output_dir and hasattr(prediction, 'conf') and prediction.conf is not None:
            conf = np.clip(prediction.conf[i], 0, 1)
            conf_16bit = (conf * 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(conf_output_dir, f"{img_name}.png"), conf_16bit)

    return saved_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for depth maps')
    parser.add_argument('--model', type=str, default='da3-large',
                        help='Model variant')
    parser.add_argument('--process-res', type=int, default=1064,
                        help='Processing resolution')
    parser.add_argument('--multi-view-batch', type=int, default=8,
                        help='Multi-view batch size')
    parser.add_argument('--use-ray-pose', action='store_true',
                        help='Use ray-based pose estimation')
    parser.add_argument('--save-confidence', action='store_true',
                        help='Save confidence maps')
    parser.add_argument('--max-depth', type=float, default=2.5,
                        help='Max depth for normalization in meters')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_paths = get_image_list(args.img_path)
    if not image_paths:
        print(f"Error: No images found at {args.img_path}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images")

    os.makedirs(args.outdir, exist_ok=True)
    conf_dir = None
    if args.save_confidence:
        conf_dir = os.path.join(os.path.dirname(args.outdir), "depth_confidence")
        os.makedirs(conf_dir, exist_ok=True)
    
    print(f"Loading model: depth-anything/{args.model.upper()}...")
    model = DepthAnything3.from_pretrained("./DA3-LARGE-1").to(device)
    model.eval()
    print("Model loaded.")

    batch_size = max(1, args.multi_view_batch)
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    all_saved = []
    with tqdm(total=len(image_paths), desc="Processing", unit="img") as pbar:
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(image_paths))
            batch = image_paths[start:end]

            try:
                saved = process_batch(model, batch, args.outdir, conf_dir, args.process_res, args.use_ray_pose, args.max_depth)
                all_saved.extend(saved)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\nError: {e}")
                pbar.update(len(batch))
        
    
    save_depth_metadata(args.outdir, args.max_depth, all_saved)

    print(f"\nDone! Metric depth maps saved to: {args.outdir}")
    print(f"Max depth normalization: {args.max_depth}m")

if __name__ == '__main__':
    main()