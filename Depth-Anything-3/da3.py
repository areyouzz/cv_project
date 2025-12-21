import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from depth_anything_3.api import DepthAnything3

parser = ArgumentParser("DA3")
parser.add_argument("--source_path", "-s", required=True, type=str)
args = parser.parse_args()

def generate_depth(input_path, output_path, visual_path):
    process_res = 756
    model_dir = "./DA3-LARGE-1.1"
    batch_size = 5

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(visual_path, exist_ok=True)
    
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(model_dir).to(device)

    exts = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
    image_paths.sort()

    for i in tqdm(range(0, len(image_paths), batch_size)):
        chunk_paths = image_paths[i: i + batch_size]
        
        prediction = model.inference(
            image=chunk_paths,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            align_to_input_ext_scale=False,
        )

        depths = prediction.depth # (B, H, W)

        for idx, depth_map in enumerate(depths):
            img_path = chunk_paths[idx]
            img_name = os.path.basename(img_path)
            basename = os.path.splitext(img_name)[0]

            # 保存为 16-bit PNG
            d_min = depth_map.min()
            d_max = depth_map.max()

            if d_max - d_min > 1e-8:
                depth_norm = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_map)

            # 映射到 0 - 65535
            depth_uint16 = (depth_norm * 65535).astype(np.uint16)

            save_path = os.path.join(output_path, basename + ".png")
            cv2.imwrite(save_path, depth_uint16)

            # 可视化
            vis_path = os.path.join(visual_path, basename + ".jpg")
            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(vis_path, depth_vis_color)
        
    print("\nDone!")
    print(f"Depth maps saved to: {output_path}")
    print(f"Visualizations saved to: {visual_path}")

if __name__ == "__main__":
    input_path = os.path.join(args.source_path, "images")
    output_path = os.path.join(args.source_path, "depths")
    visual_path = os.path.join(args.source_path, "depths_vs")

    generate_depth(input_path, output_path, visual_path)