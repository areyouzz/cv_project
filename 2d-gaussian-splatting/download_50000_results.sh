#!/bin/bash
# 50000迭代结果下载脚本

echo "="========================================
echo "2D-GS 50000迭代结果下载"
echo "="========================================

SERVER_IP=$(hostname -I | awk '{print $1}')
USER="cv25_010"

echo "服务器: $USER@$SERVER_IP"
echo ""

echo "选择下载内容:"
echo "1. 只下载50000迭代点云文件"
echo "2. 下载所有可视化结果"
echo "3. 下载30000和50000对比"
echo "4. 下载整个50000迭代目录"
read -p "选择 (1-4): " choice

case $choice in
    1)
        echo "下载50000迭代点云..."
        scp $USER@$SERVER_IP:output/my_video_50000/point_cloud/iteration_50000/point_cloud.ply ./point_cloud_50000.ply
        echo "✅ 下载完成: ./point_cloud_50000.ply"
        echo "用Meshlab查看: meshlab ./point_cloud_50000.ply"
        ;;
    2)
        echo "下载可视化结果..."
        scp -r $USER@$SERVER_IP:/datadisk/home/cv25_010/code/cv_project/2d-gaussian-splatting/output/50000_results ./50000_visualization/
        echo "✅ 下载完成"
        echo "用浏览器打开: ./50000_visualization/viewer_50000.html"
        ;;
    3)
        echo "下载对比结果..."
        mkdir -p ./comparison_30000_vs_50000
        # 下载30000迭代
        scp $USER@$SERVER_IP:output/my_video_final/point_cloud/iteration_30000/point_cloud.ply ./comparison_30000_vs_50000/point_cloud_30000.ply
        # 下载50000迭代
        scp $USER@$SERVER_IP:output/my_video_50000/point_cloud/iteration_50000/point_cloud.ply ./comparison_30000_vs_50000/point_cloud_50000.ply
        echo "✅ 对比文件下载完成"
        echo "比较命令:"
        echo "  meshlab ./comparison_30000_vs_50000/point_cloud_30000.ply"
        echo "  meshlab ./comparison_30000_vs_50000/point_cloud_50000.ply"
        ;;
    4)
        echo "下载整个50000迭代目录..."
        scp -r $USER@$SERVER_IP:/datadisk/home/cv25_010/code/cv_project/2d-gaussian-splatting/output/my_video_50000 ./2dgs_50000_complete/
        echo "✅ 下载完成"
        ;;
    *)
        echo "无效选择"
        ;;
esac

echo ""
echo "🎯 查看建议:"
echo "1. 用Meshlab查看PLY文件"
echo "2. 用浏览器打开HTML查看器"
echo "3. 对比30000和50000迭代的细节差异"
