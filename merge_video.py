import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_name, fps=5, max_images=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 确保图片按顺序排列
    images = images[:max_images]  # 只取前max_images张图片

    # 读取第一张图片来获取尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image in tqdm(images, desc="处理图片"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放视频写入器
    video.release()

    print(f"视频已保存为 {output_video_name}")

# 使用函数
image_folder = "temp_3d_vis"
output_video = "wanroad_car_gt.mp4"
create_video_from_images(image_folder, output_video, max_images=1000)