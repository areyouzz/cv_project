# video_to_2dgs_improved.py
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import shutil
import sys
import math

def extract_frames_uniform(video_path, output_dir, target_frames=300, quality=95):
    """
    从视频均匀提取帧，确保时间均匀分布
    
    Args:
        video_path: 视频文件完整路径
        output_dir: 输出目录
        target_frames: 目标帧数（最多300）
        quality: JPG质量 (1-100)
    """
    print("="*60)
    print("2D-GS 视频处理脚本 (改进版 - 均匀抽帧)")
    print("="*60)
    
    # 检查视频文件
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        
        # 尝试查找视频
        print("\n🔍 尝试查找视频文件...")
        possible_locations = [
            "/datadisk/home/cv25_010/code/cv_project/item_1.mp4",
            "/home/cv25_010/code/cv_project/item_1.mp4",
            "item_1.mp4",
            "/home/cv25_010/cv_project/item_1.mp4",
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                video_path = Path(loc)
                print(f"✅ 找到视频: {video_path}")
                break
        else:
            print("❌ 没有找到视频文件")
            print("请将视频文件放在以下位置之一:")
            for loc in possible_locations:
                print(f"  - {loc}")
            return None, None
    
    print(f"📹 视频文件: {video_path}")
    
    # 创建Blender格式目录结构
    base_dir = Path(output_dir)
    train_dir = base_dir / "train"  # 必须叫train，不是images
    test_dir = base_dir / "test"    # 测试图片目录
    
    for dir_path in [train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件")
        return None, None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n📊 视频信息:")
    print(f"  FPS: {fps:.2f}")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}秒")
    print(f"  目标帧数: {target_frames}")
    
    # 计算均匀采样间隔
    frame_interval = max(1, total_frames // target_frames)
    actual_target = min(target_frames, total_frames // frame_interval)
    
    print(f"  计算帧间隔: {frame_interval}")
    print(f"  预计提取: {actual_target} 张")
    
    # 计算采样帧索引
    frame_indices = np.linspace(0, total_frames-1, actual_target, dtype=int)
    
    saved_count = 0
    saved_files = []
    
    # 进度条
    pbar = tqdm(total=actual_target, desc="均匀提取帧")
    
    for idx, frame_idx in enumerate(frame_indices):
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # 生成文件名 - 按照Blender格式: r_{数字}.png
        frame_filename = f"r_{saved_count:04d}.png"
        output_path = train_dir / frame_filename
        
        # 保存为PNG（Blender格式用PNG）
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 保存
        img = Image.fromarray(frame_rgb)
        img.save(output_path, "PNG", optimize=True)
        
        saved_files.append({
            "index": saved_count,
            "filename": frame_filename,
            "path": str(output_path),
            "original_frame": frame_idx,
            "time_sec": frame_idx / fps
        })
        saved_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    if saved_count == 0:
        print("❌ 没有提取到任何帧")
        return None, None
    
    print(f"\n✅ 提取完成:")
    print(f"  均匀提取了 {saved_count} 张PNG图片")
    print(f"  保存在: {train_dir}")
    
    # 显示时间分布
    if saved_files:
        times = [f["time_sec"] for f in saved_files]
        print(f"  时间范围: {times[0]:.1f}秒 到 {times[-1]:.1f}秒")
        print(f"  平均间隔: {(times[-1]-times[0])/(len(times)-1):.2f}秒")
        
        # 显示第一张图片信息
        first_file = train_dir / saved_files[0]["filename"]
        img = Image.open(first_file)
        print(f"  图片尺寸: {img.size[0]}x{img.size[1]}")
        print(f"  格式: {img.format}")
    
    return base_dir, saved_files

def create_precise_camera_poses(base_dir, image_files, camera_angle_x=0.6911112070083618, 
                               radius=3.0, height=1.5, look_at=(0, 0, 0)):
    """
    创建精确的圆形相机轨迹
    
    Args:
        base_dir: 基础目录
        image_files: 图片文件列表
        camera_angle_x: 相机水平视角（弧度）
        radius: 相机轨道半径
        height: 相机高度
        look_at: 看向的点坐标
    """
    print("\n📄 创建精确相机位姿...")
    
    base_dir = Path(base_dir)
    train_dir = base_dir / "train"
    
    if not image_files:
        print("❌ 没有图片文件")
        return False
    
    # 获取图片尺寸
    first_file = train_dir / image_files[0]["filename"]
    img = Image.open(first_file)
    width, height_px = img.size
    
    print(f"  图片尺寸: {width}x{height_px}")
    print(f"  相机角度: {camera_angle_x} 弧度 ({np.degrees(camera_angle_x):.1f}°)")
    print(f"  轨道半径: {radius}")
    print(f"  相机高度: {height}")
    print(f"  看向点: {look_at}")
    print(f"  总帧数: {len(image_files)}")
    
    # 创建transforms_train.json
    transforms_train = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    look_at = np.array(look_at)
    
    # 为每张训练图片创建相机位姿
    for i, img_info in enumerate(image_files):
        # 计算角度 (均匀分布0-360度)
        angle = 2 * np.pi * i / len(image_files)
        
        # 相机位置 (圆形轨迹)
        x = radius * np.cos(angle)
        y = height
        z = radius * np.sin(angle)
        
        # 相机位置向量
        eye = np.array([x, y, z])
        
        # 计算看向目标的变换矩阵
        # 1. 计算前向向量 (从相机指向目标)
        forward = look_at - eye
        forward = forward / np.linalg.norm(forward)
        
        # 2. 初始上向量
        world_up = np.array([0, 1, 0])
        
        # 3. 计算右向量
        right = np.cross(forward, world_up)
        # 如果右向量长度为0，说明forward和world_up平行
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])  # 使用默认右向量
        right = right / np.linalg.norm(right)
        
        # 4. 重新计算上向量以确保正交
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 创建4x4相机到世界变换矩阵
        # 注意：在NeRF/Blender格式中，这是相机到世界的变换
        transform = np.eye(4)
        transform[:3, 0] = right      # 右向量
        transform[:3, 1] = up         # 上向量  
        transform[:3, 2] = -forward   # 前向量（取反，因为相机坐标系z向前）
        transform[:3, 3] = eye        # 位置
        
        frame = {
            "file_path": f"./train/{img_info['filename'].replace('.png', '')}",
            "rotation": 0.012566370614359171,  # 标准值
            "transform_matrix": transform.tolist()
        }
        transforms_train["frames"].append(frame)
        
        # 调试：显示前几个相机的位置
        if i < 3:
            print(f"  相机 {i}: 位置({x:.2f}, {y:.2f}, {z:.2f}), 角度{np.degrees(angle):.1f}°")
    
    # 保存transforms_train.json
    transforms_train_file = base_dir / "transforms_train.json"
    with open(transforms_train_file, 'w') as f:
        json.dump(transforms_train, f, indent=2)
    
    print(f"✅ 创建: {transforms_train_file}")
    print(f"  训练帧数: {len(transforms_train['frames'])}")
    
    # 创建transforms_test.json（均匀选择测试帧）
    transforms_test = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    test_dir = base_dir / "test"
    test_dir.mkdir(exist_ok=True)
    
    # 均匀选择测试帧 (大约10%的训练帧)
    num_test = max(5, len(image_files) // 10)
    test_indices = np.linspace(0, len(image_files)-1, num_test, dtype=int)
    
    print(f"  选择 {num_test} 张测试帧: {list(test_indices)}")
    
    for idx in test_indices:
        if idx < len(image_files):
            img_info = image_files[idx]
            
            # 使用稍微不同的角度（偏移10度）
            angle_offset = np.radians(10)
            angle = 2 * np.pi * idx / len(image_files) + angle_offset
            
            x = radius * np.cos(angle)
            y = height
            z = radius * np.sin(angle)
            
            eye = np.array([x, y, z])
            forward = look_at - eye
            forward = forward / np.linalg.norm(forward)
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1, 0, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            transform = np.eye(4)
            transform[:3, 0] = right
            transform[:3, 1] = up
            transform[:3, 2] = -forward
            transform[:3, 3] = eye
            
            frame = {
                "file_path": f"./test/{img_info['filename'].replace('.png', '')}",
                "rotation": 0.012566370614359171,
                "transform_matrix": transform.tolist()
            }
            transforms_test["frames"].append(frame)
            
            # 复制图片到test目录
            src = train_dir / img_info["filename"]
            dst = test_dir / img_info["filename"]
            shutil.copy2(src, dst)
    
    transforms_test_file = base_dir / "transforms_test.json"
    with open(transforms_test_file, 'w') as f:
        json.dump(transforms_test, f, indent=2)
    
    print(f"✅ 创建: {transforms_test_file}")
    print(f"  测试帧数: {len(transforms_test['frames'])}")
    
    # 创建相机轨迹可视化
    create_camera_trajectory_visualization(transforms_train, base_dir)
    
    return True

def create_camera_trajectory_visualization(transforms_data, output_dir):
    """创建相机轨迹可视化"""
    print("\n📈 创建相机轨迹可视化...")
    
    # 提取所有相机位置
    positions = []
    for frame in transforms_data["frames"]:
        transform = np.array(frame["transform_matrix"])
        position = transform[:3, 3]
        positions.append(position)
    
    positions = np.array(positions)
    
    # 创建3D轨迹图
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.6, linewidth=1)
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=range(len(positions)), 
               cmap='viridis', s=20, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('相机3D轨迹')
    ax1.grid(True, alpha=0.3)
    
    # XY投影
    ax2 = fig.add_subplot(222)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6, linewidth=1)
    ax2.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)), 
               cmap='viridis', s=20, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY平面投影')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # XZ投影
    ax3 = fig.add_subplot(223)
    ax3.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.6, linewidth=1)
    ax3.scatter(positions[:, 0], positions[:, 2], c=range(len(positions)), 
               cmap='viridis', s=20, alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ平面投影')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # 高度变化
    ax4 = fig.add_subplot(224)
    frames = range(len(positions))
    ax4.plot(frames, positions[:, 1], 'g-', linewidth=2)
    ax4.set_xlabel('帧序号')
    ax4.set_ylabel('Y (高度)')
    ax4.set_title('相机高度变化')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'相机轨迹分析 ({len(positions)}个相机位置)', fontsize=14)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "camera_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 相机轨迹图: {output_path}")
    
    # 保存轨迹数据
    traj_data = {
        "num_cameras": len(positions),
        "positions": positions.tolist(),
        "radius_avg": np.mean(np.sqrt(positions[:, 0]**2 + positions[:, 2]**2)),
        "height_avg": np.mean(positions[:, 1]),
        "height_std": np.std(positions[:, 1])
    }
    
    with open(output_dir / "camera_trajectory.json", 'w') as f:
        json.dump(traj_data, f, indent=2)
    
    print(f"  平均轨道半径: {traj_data['radius_avg']:.2f}")
    print(f"  平均高度: {traj_data['height_avg']:.2f} ± {traj_data['height_std']:.2f}")

def analyze_video_for_best_params(video_path):
    """分析视频以确定最佳参数"""
    print("\n🔍 分析视频确定最佳参数...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  视频时长: {duration:.1f}秒")
    print(f"  总帧数: {total_frames}")
    
    # 读取几帧分析运动
    sample_frames = min(10, total_frames)
    frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    
    prev_frame = None
    motion_scores = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # 计算帧间差异
            diff = cv2.absdiff(gray, prev_frame)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        
        prev_frame = gray
    
    cap.release()
    
    if motion_scores:
        avg_motion = np.mean(motion_scores)
        print(f"  平均运动分数: {avg_motion:.1f}")
        
        # 根据运动确定目标帧数
        if avg_motion > 50:  # 快速运动
            target_frames = 300
            print(f"  检测到快速运动，推荐帧数: {target_frames}")
        elif avg_motion > 20:  # 中等运动
            target_frames = 200
            print(f"  检测到中等运动，推荐帧数: {target_frames}")
        else:  # 慢速运动
            target_frames = 150
            print(f"  检测到慢速运动，推荐帧数: {target_frames}")
    else:
        target_frames = 200
        print(f"  使用默认帧数: {target_frames}")
    
    return min(target_frames, 300)  # 最多300帧

def create_quality_report(base_dir, image_files, video_info):
    """创建质量报告"""
    print("\n📊 创建数据处理报告...")
    
    report = {
        "video_info": video_info,
        "extraction_info": {
            "total_frames_extracted": len(image_files),
            "frame_indices": [f["original_frame"] for f in image_files],
            "time_points": [f["time_sec"] for f in image_files],
            "time_span": f"{image_files[0]['time_sec']:.1f}s - {image_files[-1]['time_sec']:.1f}s"
        },
        "camera_config": {
            "num_train_frames": len(image_files),
            "camera_angle_x": 0.6911112070083618,
            "trajectory_radius": 3.0,
            "camera_height": 1.5
        },
        "quality_metrics": {
            "time_uniformity": "优" if len(image_files) > 100 else "良",
            "frame_coverage": f"{(len(image_files) / video_info['total_frames'] * 100):.1f}%",
            "recommended_iterations": 50000 if len(image_files) > 200 else 30000
        }
    }
    
    report_path = base_dir / "processing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 处理报告: {report_path}")
    
    # 生成Markdown报告
    md_report = f"""# 2D-GS 视频处理报告

## 视频信息
- 文件: {video_info['path']}
- 时长: {video_info['duration']:.1f}秒
- 总帧数: {video_info['total_frames']}
- FPS: {video_info['fps']:.2f}

## 抽帧结果
- 提取帧数: {len(image_files)}
- 时间范围: {report['extraction_info']['time_span']}
- 帧覆盖: {report['quality_metrics']['frame_coverage']}
- 时间均匀性: {report['quality_metrics']['time_uniformity']}

## 相机配置
- 训练帧: {report['camera_config']['num_train_frames']}
- 相机轨道半径: {report['camera_config']['trajectory_radius']}
- 相机高度: {report['camera_config']['camera_height']}
- 水平视角: {np.degrees(report['camera_config']['camera_angle_x']):.1f}°

## 训练建议
- 推荐迭代: {report['quality_metrics']['recommended_iterations']:,}
- 建议命令:
## 生成时间
{os.popen('date').read().strip()}
"""
    
    md_path = base_dir / "processing_report.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    print(f"📄 Markdown报告: {md_path}")
    
    return report

def main():
    """主函数"""
    # 视频文件路径
    video_path = "/datadisk/home/cv25_010/code/cv_project/2d-gaussian-splatting/kapybara.mp4"
    output_dir = "data/capybara"
    
    print("="*60)
    print("2D-Gaussian Splatting 视频处理 (改进版)")
    print("="*60)
    
    # 检查视频文件
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 分析视频确定最佳参数
    target_frames = analyze_video_for_best_params(video_path)
    
    print(f"\n🎯 目标帧数: {target_frames}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    video_info = {
        "path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration
    }
    
    # 选项
    print("\n选择处理模式:")
    print("1. 自动模式 (推荐)")
    print("2. 自定义帧数")
    
    choice = input("\n请选择 (1 或 2): ").strip()
    
    if choice == "2":
        try:
            custom_frames = int(input(f"输入目标帧数 (1-{min(500, total_frames)}): ").strip())
            target_frames = min(max(1, custom_frames), 500)
        except:
            print("使用自动模式")
    
    # 处理视频
    print(f"\n🚀 开始处理视频...")
    base_dir, image_files = extract_frames_uniform(
        video_path=video_path,
        output_dir=output_dir,
        target_frames=target_frames
    )
    
    if base_dir and image_files:
        # 创建配置文件
        success = create_precise_camera_poses(
            base_dir=base_dir,
            image_files=image_files,
            camera_angle_x=0.6911112070083618,
            radius=3.0,
            height=1.5,
            look_at=(0, 0, 0)
        )
        
        if success:
            # 创建质量报告
            report = create_quality_report(base_dir, image_files, video_info)
            
            print(f"\n" + "="*60)
            print("✅ 处理完成!")
            print("="*60)
            
            print(f"\n📁 数据集路径: {base_dir}")
            print(f"📊 提取帧数: {len(image_files)}")
            
            print(f"\n🚀 训练命令:")
            rec_iter = report['quality_metrics']['recommended_iterations']
            print(f"python train.py -s {base_dir} \\")
            print(f"  -m output/model_improved \\")
            print(f"  --iterations {rec_iter} \\")
            print(f"  --save_iterations {rec_iter//5} {rec_iter//2} {rec_iter} \\")
            print(f"  --resolution 1 \\")
            print(f"  --white_background \\")
            print(f"  --quiet")
            
            print(f"\n💡 提示: 更多帧数需要更多迭代才能获得好效果")
            print(f"       {len(image_files)}帧建议使用{rec_iter:,}次迭代")

if __name__ == "__main__":
    # 设置matplotlib后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    main()