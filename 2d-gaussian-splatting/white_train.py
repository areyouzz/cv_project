#!/usr/bin/env python3
# white_train_final.py - 完全正确的白墙训练
import os
import sys
import torch
from random import randint
import numpy as np
from pathlib import Path
import cv2
import json
import collections

# 添加原始代码路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始模块
try:
    from utils.loss_utils import l1_loss, ssim
    from gaussian_renderer import render, network_gui
    from scene import Scene, GaussianModel
    from utils.general_utils import safe_state
    import uuid
    from tqdm import tqdm
    from utils.image_utils import psnr, render_net_image
    from argparse import ArgumentParser, Namespace
    from arguments import ModelParams, PipelineParams, OptimizationParams
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_FOUND = True
    except ImportError:
        TENSORBOARD_FOUND = False
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# ========== 创建完全正确的CameraInfo ==========

def create_camera_info_class():
    """创建与原始CameraInfo完全相同的类"""
    CameraInfo = collections.namedtuple(
        "CameraInfo", 
        ["uid", "R", "T", "FovY", "FovX", "image", "image_path", "image_name", 
         "width", "height", "gt_alpha_mask", "gt_depth", "original_image", "FoVx", "FoVy"]
    )
    
    return CameraInfo

# ========== 白墙场景创建模块 ==========

class WhiteWallScene:
    """白墙场景，使用完全正确的CameraInfo"""
    
    def __init__(self, source_path, white_background=True, images="images"):
        self.source_path = Path(source_path)
        self.white_background = white_background
        self.model_path = None
        
        # 获取CameraInfo类
        self.CameraInfo = create_camera_info_class()
        
        # 创建相机
        self.train_cameras = self.create_cameras()
        self.test_cameras = []
        
        # 计算场景范围
        self.cameras_extent = self.compute_extent()
        
        # 保存路径
        self.ply_path = None
        
    def create_cameras(self):
        """为白墙场景创建相机"""
        
        # 获取图片目录
        images_dir = self.source_path / "images"
        if not images_dir.exists():
            images_dir = self.source_path / "input"
        
        # 获取所有图片
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(sorted(images_dir.glob(f"*{ext}")))
        
        if not image_files:
            raise ValueError(f"在 {images_dir} 中没有找到图片")
        
        print(f"找到 {len(image_files)} 张图片")
        
        cam_infos = []
        for i, img_path in enumerate(image_files):
            # 读取图片
            image, original_image, width, height = self.load_image(img_path)
            
            # 创建合理的相机轨迹（圆形）
            angle = 2 * np.pi * i / len(image_files)
            radius = 2.5  # 相机距离
            
            # 相机位置
            x = radius * np.sin(angle)
            y = 0.3  # 稍微抬高
            z = radius * np.cos(angle)
            
            # 看向原点
            pos = np.array([x, y, z])
            target = np.array([0, 0.1, 0])
            
            forward = target - pos
            forward = forward / np.linalg.norm(forward)
            
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 0.001:
                right = np.array([1, 0, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 旋转矩阵 (世界到相机)
            R = np.column_stack([right, up, -forward])
            
            # 平移
            T = -R.T @ pos
            
            # 计算FOV
            focal = max(width, height) * 1.2
            fovx = 2 * np.arctan(width / (2 * focal))
            fovy = 2 * np.arctan(height / (2 * focal))
            
            # 创建完全正确的CameraInfo
            cam_info = self.CameraInfo(
                uid=i,
                R=R,
                T=T,
                FovY=np.degrees(fovy),
                FovX=np.degrees(fovx),
                image=image,  # numpy数组
                image_path=str(img_path),
                image_name=img_path.name,
                width=width,
                height=height,
                gt_alpha_mask=None,  # 白墙没有alpha mask
                gt_depth=None,       # 白墙没有深度
                original_image=original_image,  # torch tensor
                FoVx=np.degrees(fovx),  # 添加FoVx属性
                FoVy=np.degrees(fovy)   # 添加FoVy属性
            )
            
            cam_infos.append(cam_info)
        
        return cam_infos
    
    def load_image(self, img_path):
        """加载并预处理图片"""
        try:
            # 使用OpenCV加载
            image = cv2.imread(str(img_path))
            if image is None:
                # 返回默认值
                width, height = 1024, 768
                return None, torch.zeros((3, height, width)), width, height
            
            # 增强白墙对比度
            image = self.enhance_white_wall(image)
            
            height, width = image.shape[:2]
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 归一化到[0, 1]
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            # 转换为torch tensor (C, H, W)
            original_image = torch.from_numpy(image_normalized).permute(2, 0, 1)
            
            return image_normalized, original_image, width, height
            
        except Exception as e:
            print(f"加载图片 {img_path} 失败: {e}")
            width, height = 1024, 768
            return None, torch.zeros((3, height, width)), width, height
    
    def enhance_white_wall(self, image):
        """增强白墙图片对比度"""
        # 1. CLAHE增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. 边缘增强
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def compute_extent(self):
        """计算场景范围"""
        if not self.train_cameras:
            return 4.0
        
        # 收集所有相机位置
        positions = []
        for cam in self.train_cameras:
            # 从旋转和平移计算位置
            R = cam.R
            T = cam.T
            # 位置 = -R^T * T
            pos = -R.T @ T
            positions.append(pos)
        
        if not positions:
            return 4.0
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        extent = np.max(distances) * 1.5  # 添加一些余量
        
        return max(extent, 4.0)
    
    def getTrainCameras(self):
        return self.train_cameras
    
    def getTestCameras(self):
        return self.test_cameras
    
    def save(self, iteration):
        """保存场景"""
        if self.model_path and hasattr(self, 'gaussians'):
            output_dir = Path(self.model_path)
            point_cloud_dir = output_dir / f"point_cloud/iteration_{iteration}"
            point_cloud_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存点云
            self.gaussians.save_ply(str(point_cloud_dir / "point_cloud.ply"))

# ========== 适配器 ==========

class SceneAdapter:
    """适配器，让WhiteWallScene可以像原Scene类一样工作"""
    
    def __init__(self, white_wall_scene, gaussians):
        self.white_wall_scene = white_wall_scene
        self.gaussians = gaussians
        
    def getTrainCameras(self):
        cameras = self.white_wall_scene.getTrainCameras()
        # 确保每个相机都有正确的属性
        for cam in cameras:
            # 添加任何缺失的属性
            if not hasattr(cam, 'FoVx'):
                cam.FoVx = cam.FovX
            if not hasattr(cam, 'FoVy'):
                cam.FoVy = cam.FovY
        return cameras
    
    def getTestCameras(self):
        cameras = self.white_wall_scene.getTestCameras()
        for cam in cameras:
            if not hasattr(cam, 'FoVx'):
                cam.FoVx = cam.FovX
            if not hasattr(cam, 'FoVy'):
                cam.FoVy = cam.FovY
        return cameras
    
    @property
    def cameras_extent(self):
        return self.white_wall_scene.cameras_extent
    
    def save(self, iteration):
        """保存高斯模型"""
        if hasattr(self.white_wall_scene, 'model_path'):
            output_dir = Path(self.white_wall_scene.model_path)
            point_cloud_dir = output_dir / f"point_cloud/iteration_{iteration}"
            point_cloud_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存点云
            self.gaussians.save_ply(str(point_cloud_dir / "point_cloud.ply"))
            print(f"保存点云到 {point_cloud_dir}/point_cloud.ply")

# ========== 主训练函数 ==========

def white_wall_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    """白墙训练函数"""
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # 创建高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 使用我们的白墙场景
    white_scene = WhiteWallScene(
        source_path=dataset.source_path,
        white_background=dataset.white_background
    )
    white_scene.model_path = dataset.model_path
    
    # 创建适配器
    scene = SceneAdapter(white_scene, gaussians)
    
    # 设置训练
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # 背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 计时器
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        
        # 更新学习率
        gaussians.update_learning_rate(iteration)
        
        # 每1000次迭代增加SH度数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # 选择随机相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        
        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 正则化
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()
        
        # 总损失
        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        
        iter_end.record()
        
        with torch.no_grad():
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()
            
            # 记录和保存
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
            
            # 训练报告
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, 
                iter_start.elapsed_time(iter_end), testing_iterations, 
                scene, render, (pipe, background)
            )
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
            
            # 致密化
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull, 
                        scene.cameras_extent, size_threshold
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration), 
                    dataset.model_path + "/chkpnt" + str(iteration) + ".pth"
                )
        
        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                        )
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):
    """准备输出和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    """训练报告"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        
                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                            
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass
                        
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        torch.cuda.empty_cache()

# ========== 主程序 ==========

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="White wall training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("=" * 60)
    print("WHITE WALL TRAINING MODE")
    print("=" * 60)
    print(f"Optimizing {args.model_path}")
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    # 启动GUI服务器并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 运行白墙训练
    white_wall_training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, 
        args.checkpoint_iterations, args.start_checkpoint
    )
    
    # 完成
    print("\n" + "=" * 60)
    print("White wall training complete!")
    print("=" * 60)