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

import os
import logging
from argparse import ArgumentParser
import shutil

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import Sam3Processor, Sam3Model


# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter with SAM3")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--prompt", default="", type=str, help="Text prompt for SAM3 segmentation")
parser.add_argument("--skip_masking", action='store_true', help="Skip SAM 3 mask generation if already done")
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

def generate_mask(input_dir, output_dir, prompt):
    print(f"\n[SAM 3] Initializing for prompt: '{prompt}'...")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    
    try:
        processor = Sam3Processor.from_pretrained("./sam3")
        model = Sam3Model.from_pretrained("./sam3").to(device)
    except Exception as e:
        print(f"Failed to load model via transformers: {e}")
        return 

    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    print(f"[SAM 3] Processing {len(image_files)} images...")

    for img_name in tqdm(image_files):
        save_name = f"{img_name}.png"
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            continue

        img_path = os.path.join(input_dir, img_name)
        try: 
            image_pil = Image.open(img_path).convert("RGB")

            # Segment using text prompt
            inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process results
            original_sizes = [(image_pil.height, image_pil.width)]

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.001, # 置信度阈值
                mask_threshold=0.5,
                target_sizes = original_sizes
            )[0] # 取第一张

            masks = results['masks']
            scores = results['scores']
            
            if len(masks) > 0:
                # 选分数最高的 mask
                best_idx = torch.argmax(scores).item()
                best_mask = masks[best_idx]
                mask_np = (best_mask.cpu().numpy() > 0).astype(np.uint8) * 255 # 0/255 uint8
            else:
                mask_np = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

            Image.fromarray(mask_np).save(save_path)

        except Exception as e:
            print(f"Error masking {img_name}: {e}")

    print("[SAM 3] Mask generation done.\n")


if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    input_images_path = os.path.join(args.source_path, "input")
    masks_path = os.path.join(args.source_path, "masks")

    ## Sam3 generate masks
    if not args.skip_masking:
        generate_mask(input_images_path, masks_path, args.prompt)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.mask_path " + masks_path + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera
    
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db"
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
