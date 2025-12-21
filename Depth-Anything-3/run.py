import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_3.api import DepthAnything3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V3')
    
    parser.add_argument('--img-path', type=str, required=True, help='Path to image or folder')
    parser.add_argument('--input-size', type=int, default=518, help='Resize size (process_res)')
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    # 直接接受 DA3 的模型名，例如: da3-large, da3-base, da3-small
    parser.add_argument('--encoder', type=str, default='da3-large', help='Model name (e.g. da3-large)')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Loading Depth Anything V3 model: {args.encoder}...")
    # 直接使用 args.encoder 作为模型名加载
    # 格式应该是 "depth-anything/DA3-LARGE" 这种，或者简单的 "da3-large" (库内部会自动处理)
    # 为了稳健，我们假设用户输入的是 "da3-large"，手动拼接前缀，或者依赖库的自动推断
    # depth_anything_3.api 通常接受 "da3-large" 这样的简写
    model = DepthAnything3(model_name=args.encoder).to(DEVICE)
    
    # 获取文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            continue
            
        # DA3 API 期望 RGB 输入，OpenCV 读取的是 BGR
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        # 推理
        prediction = model.inference(
            image=[image_rgb],
            process_res=args.input_size,
            process_res_method="upper_bound_resize",
            align_to_input_ext_scale=False
        )
        
        depth = prediction.depth[0] # 取出第一张图的深度 (H, W)
        
        # 归一化到 0-255 (可视化用)
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        else:
            depth = np.zeros_like(depth)
            
        depth = depth.astype(np.uint8)

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_16bit = (depth_norm * 65535).astype(np.uint16)
        
        # 处理输出颜色
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 保存文件名
        save_name = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        
        if args.pred_only:
            # cv2.imwrite(save_name, depth)
            cv2.imwrite(save_name, depth_16bit)
        else:
            # 拼接对比图
            # 如果尺寸因为 padding 发生微小变化，做一下 resize 对齐
            if raw_image.shape[:2] != depth.shape[:2]:
                raw_image = cv2.resize(raw_image, (depth.shape[1], depth.shape[0]))
                
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(save_name, combined_result)

    print("Done.")