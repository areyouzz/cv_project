#!/bin/bash
# 改进数据结果下载脚本

echo "="========================================
echo "2D-GS 改进数据结果下载 (迭代 20000)"
echo "="========================================

SERVER_IP=$(hostname -I | awk '{print $1}')
USER="cv25_010"

echo "服务器: $USER@$SERVER_IP"
echo ""

echo "选择下载内容:"
echo "1. 只下载点云文件"
echo "2. 下载所有可视化结果"
echo "3. 下载训练目录（包含检查点）"
echo "4. 对比新旧结果"
read -p "选择 (1-4): " choice

case $choice in
    1)
        echo "下载点云文件..."
        scp $USER@$SERVER_IP:output/my_model/point_cloud/iteration_10000/point_cloud.ply ./point_cloud_improved_20000.ply
        echo "✅ 下载完成: ./point_cloud_improved_20000.ply"
        echo "用Meshlab查看: meshlab ./point_cloud_improved_20000.ply"
        ;;
    2)
        echo "下载可视化结果..."
        scp -r $USER@$SERVER_IP:output/improved_results_20000 ./improved_visualization_20000/
        echo "✅ 下载完成"
        echo "用浏览器打开: ./improved_visualization_20000/viewer_improved_20000.html"
        ;;
    3)
        echo "下载训练目录..."
        scp -r $USER@$SERVER_IP:/datadisk/home/cv25_010/code/cv_project/2d-gaussian-splatting/output/my_model ./improved_training_20000/
        echo "✅ 下载完成"
        echo "包含检查点，可继续训练"
        ;;
    4)
        echo "下载对比结果..."
        mkdir -p ./comparison_results
        # 下载旧结果
        scp $USER@$SERVER_IP:output/my_video_final/point_cloud/iteration_30000/point_cloud.ply ./comparison_results/old_30000.ply 2>/dev/null || echo "旧结果不存在"
        # 下载新结果
        scp $USER@$SERVER_IP:output/my_model/point_cloud/iteration_10000/point_cloud.ply ./comparison_results/new_improved_20000.ply
        echo "✅ 对比文件下载完成"
        echo "比较命令:"
        echo "  meshlab ./comparison_results/old_30000.ply"
        echo "  meshlab ./comparison_results/new_improved_20000.ply"
        ;;
    *)
        echo "无效选择"
        ;;
esac

echo ""
echo "🎯 查看建议:"
echo "1. Loss=0.0478 表示训练效果很好"
echo "2. 点数越多通常表示重建越好"
echo "3. 可以用更多迭代继续优化"
