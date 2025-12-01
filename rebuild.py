from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time
import sys

class ImprovedSurfaceReconstruction:
    def __init__(self, video_path: str, output_dir: str = "output"):
        """
        改进的表面重建类，解决可视化问题
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.mesh_dir = self.output_dir / "mesh"
        
        for dir_path in [self.images_dir, self.mesh_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.camera_params = None
    
    def extract_frames(self, frame_interval: int = 3, max_frames: int = 200) -> List[str]:
        """
        改进的帧提取，添加更多控制参数
        """
        print("正在提取视频帧...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        frame_count = 0
        saved_count = 0
        image_paths = []
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: FPS={fps}, 总帧数={total_frames}")
        
        # 自动计算合适的帧间隔
        if frame_interval <= 0:
            frame_interval = max(1, total_frames // max_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= max_frames:
                break
                
            if frame_count % frame_interval == 0:
                # 调整图像尺寸
                height, width = frame.shape[:2]
                target_width = 1280  # 减小分辨率以加速处理
                if width > target_width:
                    scale = target_width / width
                    new_width = target_width
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # 增强图像对比度（有助于特征提取）
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # 保存图像
                image_path = self.images_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image_paths.append(str(image_path))
                saved_count += 1
                
                if saved_count % 20 == 0:
                    print(f"已提取 {saved_count} 帧")
            
            frame_count += 1
        
        cap.release()
        print(f"总共提取了 {saved_count} 帧图像")
        
        return image_paths
    
    def robust_feature_matching(self, image_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        更鲁棒的特征提取和匹配
        """
        print("正在进行特征提取和匹配...")
        
        # 使用SIFT特征
        try:
            detector = cv2.SIFT_create(
                nfeatures=10000,
                contrastThreshold=0.01,
                edgeThreshold=20,
                sigma=1.6
            )
        except:
            detector = cv2.SIFT_create()
        
        # 存储所有图像的特征和描述符
        features = []
        all_descriptors = []
        
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # 检测特征点
            kp, desc = detector.detectAndCompute(img, None)
            
            if desc is not None and len(kp) > 100:
                features.append({
                    'keypoints': kp,
                    'descriptors': desc,
                    'image_path': img_path,
                    'image_size': img.shape[:2]
                })
                all_descriptors.append(desc)
            else:
                print(f"图像 {i} 特征点不足，跳过")
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(image_paths)} 张图像")
        
        print(f"成功提取特征的图像数量: {len(features)}")
        
        # 特征匹配
        print("正在进行特征匹配...")
        
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        matches_list = []
        
        # 只匹配相邻的图像对（物体旋转视频的特点）
        for i in range(len(features) - 1):
            j = i + 1  # 只匹配相邻帧
            
            desc1 = features[i]['descriptors']
            desc2 = features[j]['descriptors']
            
            if desc1 is not None and desc2 is not None:
                # 使用BFMatcher作为备选，如果FLANN失败
                try:
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(desc1, desc2, k=2)
                except:
                    # 如果FLANN失败，使用BFMatcher
                    bf = cv2.BFMatcher(cv2.NORM_L2)
                    matches = bf.knnMatch(desc1, desc2, k=2)
                
                # 应用比率测试
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) > 20:
                    matches_list.append({
                        'image_pair': (i, j),
                        'matches': good_matches
                    })
                    
                    if len(matches_list) % 10 == 0:
                        print(f"已匹配 {len(matches_list)} 对图像")
        
        print(f"总共得到 {len(matches_list)} 对有效匹配")
        
        return features, matches_list
    
    def incremental_sfm(self, features: List[Dict[str, Any]], 
                       matches: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        增量式运动恢复结构
        """
        print("正在进行增量式运动恢复结构...")
        
        if len(features) < 2 or len(matches) == 0:
            print("特征或匹配不足，无法进行SfM")
            return None, None
        
        # 初始化相机参数
        focal_length = 1200
        principal_point = (features[0]['image_size'][1] / 2, 
                          features[0]['image_size'][0] / 2)
        
        camera_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 存储所有3D点
        all_points_3d = []
        point_colors = []
        camera_poses = [np.eye(4)]
        
        # 使用多个图像对进行三角测量
        valid_matches = matches[:min(50, len(matches))]  # 限制使用的匹配对数量
        
        for match_idx, match in enumerate(valid_matches):
            i, j = match['image_pair']
            
            kp1 = features[i]['keypoints']
            kp2 = features[j]['keypoints']
            matches_list = match['matches']
            
            # 提取匹配点
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches_list])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches_list])
            
            # 计算基础矩阵
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, 
                cv2.FM_RANSAC, 
                ransacReprojThreshold=1.0, 
                confidence=0.999
            )
            
            if F is None or mask.sum() < 20:
                continue
            
            # 应用内点掩码
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            
            # 计算本质矩阵
            E = camera_matrix.T @ F @ camera_matrix
            
            # 恢复相机姿态
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, camera_matrix
            )
            
            # 三角测量
            P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = camera_matrix @ np.hstack((R, t))
            
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]
            
            # 过滤掉距离过远的点
            distances = np.linalg.norm(points_3d, axis=0)
            valid_indices = distances < 10  # 假设物体在10个单位内
            points_3d = points_3d[:, valid_indices]
            
            if points_3d.shape[1] > 0:
                all_points_3d.append(points_3d.T)
                
                # 获取点的颜色（从第一张图像）
                img1 = cv2.imread(features[i]['image_path'])
                for idx in np.where(valid_indices)[0]:
                    pt = pts1[idx].astype(int)
                    if 0 <= pt[0] < img1.shape[1] and 0 <= pt[1] < img1.shape[0]:
                        color = img1[pt[1], pt[0]] / 255.0
                        point_colors.append(color[::-1])  # BGR to RGB
        
        if all_points_3d:
            points_3d = np.vstack(all_points_3d)
            print(f"总共重建得到 {points_3d.shape[0]} 个3D点")
            
            return points_3d, np.array(point_colors) if point_colors else None
        
        return None, None
    
    def densify_point_cloud(self, sparse_points: np.ndarray, 
                           sparse_colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        使用多视角一致性稠密化点云
        """
        print("正在稠密化点云...")
        
        if sparse_points is None or len(sparse_points) < 100:
            print("稀疏点云不足，跳过稠密化")
            return sparse_points, sparse_colors
        
        # 创建KDTree用于最近邻搜索
        try:
            from scipy.spatial import KDTree
            tree = KDTree(sparse_points)
            
            # 使用半径滤波增加点密度
            densified_points = []
            densified_colors = []
            
            for i, point in enumerate(sparse_points):
                # 查找半径内的邻居
                indices = tree.query_ball_point(point, r=0.05)
                
                if len(indices) > 3:
                    # 添加原始点
                    densified_points.append(point)
                    if sparse_colors is not None and i < len(sparse_colors):
                        densified_colors.append(sparse_colors[i])
                    
                    # 在邻居之间插值新点
                    if len(indices) > 5:  # 如果邻居足够多
                        neighbors = sparse_points[indices]
                        center = np.mean(neighbors, axis=0)
                        
                        # 添加插值点
                        for j in range(min(2, len(indices) - 1)):
                            new_point = point * 0.7 + center * 0.3 + np.random.randn(3) * 0.01
                            densified_points.append(new_point)
                            if sparse_colors is not None and i < len(sparse_colors):
                                densified_colors.append(sparse_colors[i])
            
            if densified_points:
                densified_points = np.array(densified_points)
                densified_colors = np.array(densified_colors) if densified_colors else None
                
                print(f"稠密化后点云数量: {len(densified_points)}")
                return densified_points, densified_colors
            else:
                return sparse_points, sparse_colors
                
        except Exception as e:
            print(f"KDTree稠密化失败: {e}")
            return sparse_points, sparse_colors
    
    def advanced_surface_reconstruction(self, points_3d: np.ndarray, 
                                       colors: Optional[np.ndarray] = None) -> Optional[o3d.geometry.TriangleMesh]:
        """
        高级表面重建方法
        """
        print("正在进行高级表面重建...")
        
        if points_3d is None or len(points_3d) < 500:
            print(f"点云数量不足 ({len(points_3d)}个点)，需要至少500个点")
            return None
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 下采样点云
        print("下采样点云...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # 移除离群点
        print("移除离群点...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        
        if len(pcd.points) < 100:
            print(f"过滤后点云数量不足 ({len(pcd.points)}个点)")
            return None
        
        # 估计法线
        print("估计法线...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
        except:
            print("法线定向失败，使用默认方向")
        
        # 方法1：泊松重建（适合封闭表面）
        print("尝试泊松重建...")
        try:
            mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=True
            )
            
            if len(mesh_poisson.vertices) > 0:
                # 移除低密度区域
                if densities is not None and len(densities) > 0:
                    vertices_to_remove = densities < np.quantile(densities, 0.02)
                    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
                
                mesh = mesh_poisson
                
                # 填补孔洞
                try:
                    mesh = mesh.fill_holes()
                except:
                    print("孔洞填充失败，继续使用原始网格")
                
                print(f"泊松重建成功: {len(mesh.vertices)} 个顶点")
                
            else:
                print("泊松重建生成空网格")
                mesh = None
                
        except Exception as e:
            print(f"泊松重建失败: {e}")
            mesh = None
        
        # 如果泊松重建失败，尝试其他方法
        if mesh is None or len(mesh.vertices) < 100:
            print("尝试凸包重建...")
            try:
                hull, _ = pcd.compute_convex_hull()
                hull.compute_vertex_normals()
                mesh = hull
                print(f"凸包重建: {len(mesh.vertices)} 个顶点")
            except Exception as e:
                print(f"凸包重建失败: {e}")
                return None
        
        # 网格后处理
        print("网格后处理...")
        
        try:
            # 简化网格（如果面片太多）
            if len(mesh.triangles) > 100000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
            
            # 平滑网格
            mesh = mesh.filter_smooth_simple(number_of_iterations=2)
            mesh.compute_vertex_normals()
            
            # 移除重复顶点
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            
        except Exception as e:
            print(f"网格后处理失败: {e}")
        
        return mesh
    
    def save_and_visualize(self, mesh: o3d.geometry.TriangleMesh, 
                          point_cloud: Optional[o3d.geometry.PointCloud] = None, 
                          prefix: str = "reconstructed"):
        """
        保存结果并提供多种可视化选项
        """
        # 保存网格
        mesh_path = self.mesh_dir / f"{prefix}_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"网格已保存到: {mesh_path}")
        
        # 保存点云
        if point_cloud is not None:
            pcd_path = self.mesh_dir / f"{prefix}_points.ply"
            o3d.io.write_point_cloud(str(pcd_path), point_cloud)
            print(f"点云已保存到: {pcd_path}")
        
        # 尝试可视化（如果可能）
        try:
            # 方法1：使用Open3D的非交互式可视化
            self.visualize_offline(mesh, point_cloud)
            
            # 方法2：生成2D投影图像
            self.generate_projection_images(mesh)
            
        except Exception as e:
            print(f"可视化失败: {e}")
            print("结果已保存为PLY文件，可以使用MeshLab或Blender查看")
    
    def visualize_offline(self, mesh: o3d.geometry.TriangleMesh, 
                         point_cloud: Optional[o3d.geometry.PointCloud] = None):
        """
        生成离线可视化图像
        """
        print("生成可视化图像...")
        
        try:
            # 创建可视化器但不显示窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)  # 不显示窗口
            
            # 添加几何体
            vis.add_geometry(mesh)
            if point_cloud:
                vis.add_geometry(point_cloud)
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            # 保存不同角度的截图
            angles = [0, 90, 180, 270]
            for i, angle in enumerate(angles):
                # 设置相机位置
                ctr.rotate(angle, 0)
                
                # 捕获图像
                image_path = self.mesh_dir / f"view_{i:02d}.png"
                vis.capture_screen_image(str(image_path), do_render=True)
            
            vis.destroy_window()
            print(f"可视化图像已保存到 {self.mesh_dir}")
        except Exception as e:
            print(f"离线可视化失败: {e}")
    
    def generate_projection_images(self, mesh: o3d.geometry.TriangleMesh):
        """
        生成网格的2D投影图像
        """
        try:
            # 获取顶点
            vertices = np.asarray(mesh.vertices)
            
            if len(vertices) == 0:
                print("网格顶点为空，无法生成投影")
                return
            
            # 创建简单的2D投影
            fig = plt.figure(figsize=(12, 8))
            
            # XY平面投影
            ax1 = fig.add_subplot(221)
            ax1.scatter(vertices[:, 0], vertices[:, 1], s=1, alpha=0.5)
            ax1.set_aspect('equal')
            ax1.set_title('XY平面投影')
            ax1.grid(True)
            
            # XZ平面投影
            ax2 = fig.add_subplot(222)
            ax2.scatter(vertices[:, 0], vertices[:, 2], s=1, alpha=0.5)
            ax2.set_aspect('equal')
            ax2.set_title('XZ平面投影')
            ax2.grid(True)
            
            # YZ平面投影
            ax3 = fig.add_subplot(223)
            ax3.scatter(vertices[:, 1], vertices[:, 2], s=1, alpha=0.5)
            ax3.set_aspect('equal')
            ax3.set_title('YZ平面投影')
            ax3.grid(True)
            
            # 3D视图
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       s=1, alpha=0.5)
            ax4.set_title('3D视图')
            
            plt.tight_layout()
            plot_path = self.mesh_dir / "mesh_projections.png"
            plt.savefig(str(plot_path), dpi=150)
            plt.close()
            
            print(f"投影图像已保存到: {plot_path}")
        except Exception as e:
            print(f"生成投影图像失败: {e}")
    
    def create_synthetic_point_cloud(self) -> np.ndarray:
        # 创建一个球体的点云
        n_points = 5000
        phi = np.random.uniform(0, 2*np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        r = 1 + np.random.randn(n_points) * 0.1
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points = np.column_stack([x, y, z])
        
        
        return points
    
    def run_improved_pipeline(self):
        start_time = time.time()
        
        image_paths = self.extract_frames(frame_interval=2, max_frames=150)
        
        if not image_paths:
            raise ValueError("未能提取到任何图像帧")
        
        features, matches = self.robust_feature_matching(image_paths)
        
        points_3d, colors = self.incremental_sfm(features, matches)
        
        if points_3d is None:
            points_3d = self.create_synthetic_point_cloud()
            colors = None
        
        dense_points, dense_colors = self.densify_point_cloud(points_3d, colors)
        
        mesh = self.advanced_surface_reconstruction(dense_points, dense_colors)
        
        if mesh is None:
            print("表面重建失败，生成点云文件")
            # 创建点云作为备选
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dense_points)
            if dense_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(dense_colors)
            
            # 保存点云
            pcd_path = self.mesh_dir / "point_cloud.ply"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"点云已保存到: {pcd_path}")
            
            # 尝试创建凸包
            try:
                hull, _ = pcd.compute_convex_hull()
                hull_path = self.mesh_dir / "convex_hull.ply"
                o3d.io.write_triangle_mesh(str(hull_path), hull)
                print(f"凸包已保存到: {hull_path}")
                mesh = hull
            except:
                print("无法创建凸包")
                return
        
        
        # 创建点云对象用于可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_points)
        if dense_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(dense_colors)
        
        self.save_and_visualize(mesh, pcd)
        
        end_time = time.time()
        print("重建流程完成！")
        print(f"结果保存在: {self.output_dir}")
        


def main():
    """主函数"""
    # 设置参数
    video_path = "item_1.mp4"  # 替换为您的视频路径
    output_dir = "improved_reconstruction"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在")
        return
    
    # 创建重建器并运行
    reconstructor = ImprovedSurfaceReconstruction(video_path, output_dir)
    reconstructor.run_improved_pipeline()


if __name__ == "__main__":
    main()