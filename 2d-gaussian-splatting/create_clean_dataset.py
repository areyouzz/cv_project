# create transparent pictures
# prince_clean

import os
from PIL import Image
import numpy as np

source_base = "./data/prince_clean"
input_dir = os.path.join(source_base, "input")
mask_dir = os.path.join(source_base, "masks")
output_image_dir = os.path.join(source_base, "transparent")  

os.makedirs(output_image_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, img_name)
        mask_path = os.path.join(mask_dir, f"{img_name}.png")  
        
        original = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  
        
        rgba = original.convert('RGBA')
        rgba.putalpha(mask)
        
        # 直接保存成透明底
        output_path = os.path.join(output_image_dir, f"{os.path.splitext(img_name)[0]}.png")
        rgba.save(output_path, "PNG")
        print(f"Saved transparent: {output_path}")

print("save path", output_image_dir)
print("Done.")