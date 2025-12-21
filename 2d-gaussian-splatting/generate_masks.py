import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import Sam3Processor, Sam3Model

parser = ArgumentParser("Sam3 Masks")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--prompt", default="", type=str, help="Text prompt for SAM3 segmentation")
args = parser.parse_args()

def generate_mask(input_dir, output_dir, prompt):
    print(f"\n[SAM 3] Initializing for prompt: '{prompt}'...")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = Sam3Processor.from_pretrained("./sam3")
        model = Sam3Model.from_pretrained("./sam3").to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
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
            original_size = [(image_pil.height, image_pil.width)]

            results = processor.post_process_instance_segmentation(
                outputs, threshold=0.01, 
                mask_threshold=0.5,
                target_sizes = original_size
            )[0]

            masks = results['masks']
            scores = results['scores']

            if len(masks) > 0:
                best_idx = torch.argmax(scores).item()
                best_mask = masks[best_idx]
                mask_np = (best_mask.cpu().numpy() > 0).astype(np.uint8) * 255
            else:
                mask_np = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

            Image.fromarray(mask_np).save(save_path)
        
        except Exception as e:
            print(f"Error masking {img_name}: {e}")

    print("[SAM 3] Mask generation done.\n")

# input_images_path = os.path.join(args.source_path, "input")
input_images_path = os.path.join(args.source_path, "images")
masks_path = os.path.join(args.source_path, "masks")

generate_mask(input_images_path, masks_path, args.prompt)

print("Done.")