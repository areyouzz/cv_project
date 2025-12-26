import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import Sam3Processor, Sam3Model

parser = ArgumentParser()
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--prompt", default="white table surface", type=str, 
                    help="Text prompt for segmentation")
parser.add_argument("--output_folder", default="masks", type=str,
                    help="Output folder name")
parser.add_argument("--sam3_path", default="./sam3", type=str,
                    help="Path to SAM3 checkpoint")
args = parser.parse_args()

def generate_masks(input_dir, output_dir, prompt, sam3_path):
    print(f"\n[SAM 3] Generating table masks with prompt: '{prompt}'...")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = Sam3Processor.from_pretrained(sam3_path)
        model = Sam3Model.from_pretrained(sam3_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(valid_exts)])

    print(f"[SAM 3] Processing {len(image_files)} images...")

    for img_name in tqdm(image_files):
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}.png")

        if os.path.exists(save_path):
            continue

        img_path = os.path.join(input_dir, img_name)
        try:
            image_pil = Image.open(img_path).convert("RGB")

            # Segment
            inputs = processor(images=image_pil,
                               text=prompt,
                               return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            original_size = [(image_pil.height, image_pil.width)]

            results = processor.post_process_instance_segmentation(
                outputs, 
                threshold=0.01,
                mask_threshold=0.5,
                target_sizes=original_size
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
            print(f"\nError masking {img_name}: {e}")

    print(f"[SAM 3] Table masks saved to: {output_dir}\n")

if __name__ == "__main__":
    input_images_path = os.path.join(args.source_path, "images")
    output_masks_path = os.path.join(args.source_path, args.output_folder)
    generate_masks(input_images_path, output_masks_path, args.prompt, args.sam3_path)
    print("Done.")
