# kitti_3d_vis.py
 
 
from __future__ import print_function
 
import os
import sys
import cv2
import random
import os.path
import shutil
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from kitti_util import *
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import matplotlib
from scipy.spatial.transform import Rotation as R
import matplotlib.patches as patches
from tqdm import tqdm

# 设置支持中文的字体
# matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_boxs(label_path, calib_path):
    lines = [line.rstrip() for line in open(label_path)]
    objects = [Object3d(line) for line in lines]
    calib = Calibration(calib_path)
    boxes = []
    for obj in objects:
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P) # 获取3D框-图像(8*3)、3D框-相机坐标系(8*3)
        boxes.append(box3d_pts_2d)
    return boxes
    

def read_pcd(file_path):
    with open(file_path, 'rb') as f:
        # 跳过头部信息
        for i in range(11):
            f.readline()
        
        # 读取点云数据
        points = []
        for line in f:
            x, y, z = map(float, line.decode().strip().split()[:3])
            points.append([x, y, z])
    
    return np.array(points)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



def inverse_transform(camera_bbox, camera_k_matrix):
    """
    Inverse the transformation process to recover the 3D coordinates in sensor frame
    from the 2D bounding box in camera image.

    Parameters
    ----------
    camera_bbox : np.ndarray
        Bounding box coordinates in camera image, shape (8, 3).
    camera_k_matrix : np.ndarray
        Intrinsic matrix of the camera, shape (3, 3).

    Returns
    -------
    cords_x_y_z : np.ndarray
        3D coordinates in sensor frame, shape (3, 8).
    """

    camera_bbox = camera_bbox[:8]
    # 将输入的列表转换为numpy数组
    camera_bbox = np.array(camera_bbox)
    # Step 1: Transform camera_bbox back to bbox
    new_x = camera_bbox[:, 0].reshape(8, 1)
    new_y = camera_bbox[:, 1].reshape(8, 1)
    new_z = camera_bbox[:, 2].reshape(8, 1)

    bbox = np.concatenate([new_x * new_z, new_y * new_z, new_z], axis=1)
    bbox = np.transpose(bbox)  # Shape: (3, 8)

    # Step 2: Recover cords_y_minus_z_x from bbox
    cords_y_minus_z_x = np.linalg.solve(camera_k_matrix, bbox)

    # Step 3: Recover cords_x_y_z from cords_y_minus_z_x
    cords_x_y_z = np.zeros((3, 8))
    cords_x_y_z[0, :] = cords_y_minus_z_x[2, :]
    cords_x_y_z[1, :] = cords_y_minus_z_x[0, :]
    cords_x_y_z[2, :] = -cords_y_minus_z_x[1, :]

    return cords_x_y_z

def x_to_world_transformation(location, rotation):
    """
    Get the transformation matrix from sensor coordinates to world coordinate.
    
    Parameters
    ----------
    location : list or np.array
        The location of the sensor [x, y, z]
    rotation : list or np.array
        The rotation of the sensor [roll, yaw, pitch] in degrees

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    # Convert rotation to radians
    rotation = np.radians(rotation)
    
    # Calculate trigonometric functions
    c_y = np.cos(rotation[1])
    s_y = np.sin(rotation[1])
    c_r = np.cos(rotation[0])
    s_r = np.sin(rotation[0])
    c_p = np.cos(rotation[2])
    s_p = np.sin(rotation[2])

    matrix = np.identity(4)
    
    # Translation matrix
    matrix[0, 3] = location[0]
    matrix[1, 3] = location[1]
    matrix[2, 3] = location[2]

    # Rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def sensors2world(data, camera_param_key, cords_x_y_z):
    camera = data[camera_param_key]
    cameralocation = camera['cords'][:3]
    camerarotation = camera['cords'][3:]

    transform_matrix = x_to_world_transformation(cameralocation, camerarotation)

    # 将点转换为齐次坐标
    cords_x_y_z_1 = np.vstack((cords_x_y_z, np.ones((1, 8))))

    # 执行坐标转换
    world_cord = np.dot(transform_matrix, cords_x_y_z_1)

    # 如果需要，可以将结果转换回非齐次坐标
    # world_cord = world_cord[:3, :] / world_cord[3, :]
    return world_cord

def visualize_bev(data, bev, vehicle_positions, camera_3d_boxes_pred, camera_3d_boxes_gt, lanes_3d, ego_position, camera_params, camera_names, resolution=0.1, x_range=(-40, 40), y_range=(-40, 40), save_path='bev_visualization.png'):
    fig, ax = plt.subplots(figsize=(14, 10))  # 创建图形对象和轴对象
    im = ax.imshow(bev, cmap='gray', origin='lower')
    ax.set_title('Bird\'s Eye View (BEV)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 创建一个新的轴对象用于放置颜色条
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax, label='Depth')  # 将颜色条放在右侧，并保持垂直
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_size = int((x_max - x_min) / resolution)
    y_size = int((y_max - y_min) / resolution)

     #绘制3d车道线
    for lane in lanes_3d:
        x_coords = []
        y_coords = []
        for point in lane:
            x = (point[0] - ego_position[0]) / resolution + x_size // 2
            y = (point[1] - ego_position[1]) / resolution + y_size // 2

            # 以图像中心为原点进行水平和垂直翻转
            # x = x_size - x
            y = y_size - y
            
            if 0 <= x < x_size and 0 <= y < y_size:
                x_coords.append(x)
                y_coords.append(y)
        
        if x_coords and y_coords:
            ax.plot(x_coords, y_coords, c='blue', linewidth=2, label='lane_gt' if lane == lanes_3d[0] else None)
    
    # # 绘制车辆位置
    # for vehicle_id, vehicle_info in data['vehicles'].items():
    #     pos = vehicle_info['location'][:2]  # 只取x和y坐标
    #     rotation = vehicle_info['angle'][2]  # 假设这是绕z轴的旋转（偏航角）
    #     extent = np.array(vehicle_info['extent'][:2]) * 2

    #     # 转换到BEV坐标系
    #     x = (pos[0] - ego_position[0]) / resolution + x_size // 2
    #     y = (pos[1] - ego_position[1]) / resolution + y_size // 2
    #     # 以图像中心为原点进行水平和垂直翻转
    #     x = x_size - x
    #     y = y_size - y

    #     if 0 <= x < x_size and 0 <= y < y_size:
    #         # 创建一个旋转的矩形
    #         width = extent[0] / resolution
    #         height = extent[1] / resolution
    #         rect = patches.Rectangle((x-width/2, y-height/2), width, height, 
    #                                  angle=np.degrees(rotation), 
    #                                  fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)
    
    # 绘制摄像机3D边界框
    car_pred_label_added = False
    for camera_r3dbbx, boxes in camera_3d_boxes_pred.items():
        camera_num = camera_r3dbbx.split('_')[1]
        camera_param_key = f'camera{camera_num}'
        
        if camera_param_key not in camera_params:
            print(f"警告：未找到 {camera_param_key} 的参数")
            continue

        intrinsic = np.array(camera_params[camera_param_key]['intrinsic'])
        extrinsic = np.array(camera_params[camera_param_key]['extrinsic'])
        
        for box in boxes:
            
            # 直接还原到摄像机坐标系
            points_camera = inverse_transform(box, intrinsic)

            
            # 从摄像机坐标转换到雷达坐标
            # points_lidar_2 = camera_to_lidar(points_camera, extrinsic)
            points_lidar = sensors2world(data, camera_param_key, points_camera)
            
            # 转换到雷达坐标系
            # 提取x和y坐标
            x_coords = points_lidar[0][:8]  # 前8个元素是x坐标
            y_coords = points_lidar[1][:8]  # 前8个元素是y坐标
            
            points_bev = np.zeros((8, 2))
            points_bev[:, 0] = (np.array(x_coords) - ego_position[0]) / resolution + x_size // 2
            points_bev[:, 1] = (np.array(y_coords) - ego_position[1]) / resolution + y_size // 2

            # 以图像中心为原点进行水平和垂直翻转
            # points_bev[:, 0] = x_size - points_bev[:, 0]
            points_bev[:, 1] = y_size - points_bev[:, 1]

            # 绘制底部矩形
            for i in range(4):
                if not car_pred_label_added:
                    ax.plot([points_bev[i, 0], points_bev[(i+1)%4, 0]], 
                            [points_bev[i, 1], points_bev[(i+1)%4, 1]], 'g-', linewidth=2, label='car_pred')
                    car_pred_label_added = True
                else:
                    ax.plot([points_bev[i, 0], points_bev[(i+1)%4, 0]], 
                            [points_bev[i, 1], points_bev[(i+1)%4, 1]], 'g-', linewidth=2)

    # 绘制真实3d边界框
    car_gt_label_added = False            
    for camera_r3dbbx, boxes in camera_3d_boxes_gt.items():
        camera_num = camera_r3dbbx.split('_')[1]
        camera_param_key = f'camera{camera_num}'
        
        if camera_param_key not in camera_params:
            print(f"警告：未找到 {camera_param_key} 的参数")
            continue

        intrinsic = np.array(camera_params[camera_param_key]['intrinsic'])
        extrinsic = np.array(camera_params[camera_param_key]['extrinsic'])

        for box in boxes:
            # 直接还原到摄像机坐标系
            points_camera = inverse_transform(box, intrinsic)

            # 从摄像机坐标转换到雷达坐标
            # points_lidar_2 = camera_to_lidar(points_camera, extrinsic)
            points_lidar = sensors2world(data, camera_param_key, points_camera)
            
            # 转换到雷达坐标系
            # 提取x和y坐标
            x_coords = points_lidar[0][:8]  # 前8个元素是x坐标
            y_coords = points_lidar[1][:8]  # 前8个元素是y坐标
            
            points_bev = np.zeros((8, 2))
            points_bev[:, 0] = (np.array(x_coords) - ego_position[0]) / resolution + x_size // 2
            points_bev[:, 1] = (np.array(y_coords) - ego_position[1]) / resolution + y_size // 2

            # 以图像中心为原点进行水平和垂直翻转
            # points_bev[:, 0] = x_size - points_bev[:, 0]
            points_bev[:, 1] = y_size - points_bev[:, 1]

            # 绘制底部矩形
            for i in range(4):
                if not car_gt_label_added:
                    ax.plot([points_bev[i, 0], points_bev[(i+1)%4, 0]], 
                            [points_bev[i, 1], points_bev[(i+1)%4, 1]], 'r-', linewidth=2, label='car_gt')
                    car_gt_label_added = True
                else:
                    ax.plot([points_bev[i, 0], points_bev[(i+1)%4, 0]], 
                            [points_bev[i, 1], points_bev[(i+1)%4, 1]], 'r-', linewidth=2)
    
    # 绘制自车位置（始终在图像中心）
    ax.scatter(x_size // 2, y_size // 2, c='blue', marker='o', s=100, label='Ego Vehicle')
    
    # 添加摄像机点云图例
    for camera_name, color in zip(camera_names, plt.cm.rainbow(np.linspace(0, 1, len(camera_names)))):
        ax.plot([], [], color=color, label=camera_name, linewidth=5)

    # 将所有图例放在图片外面的右侧
    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)
    
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图片时包含图例和颜色条，并提高分辨率
    plt.close()
    print(f"Image saved to: {save_path}")

def lidar_to_camera_pixel(points_lidar, data, camera_param_key, intrinsic, extrinsic):
    camera = data[camera_param_key]
    camera_location = camera['cords'][:3]
    camera_rotation = camera['cords'][3:]
    
    # 世界坐标到相机坐标的转换矩阵
    world_to_camera = np.linalg.inv(extrinsic)
    
    points_lidar_homogeneous = np.hstack((points_lidar, np.ones((points_lidar.shape[0], 1))))
    
    # 将点从世界坐标转换到相机坐标
    points_camera = np.dot(world_to_camera, points_lidar_homogeneous.T)
    
    # 调整坐标系以匹配 inverse_transform 函数
    points_camera_adjusted = np.zeros_like(points_camera[:3, :])
    points_camera_adjusted[0, :] = points_camera[1, :]
    points_camera_adjusted[1, :] = -points_camera[2, :]
    points_camera_adjusted[2, :] = points_camera[0, :]
    
    # 视锥体裁剪
    fov_x = np.deg2rad(90)  # 假设水平视场角为90度
    fov_y = np.deg2rad(60)  # 假设垂直视场角为60度
    
    in_fov = (np.abs(points_camera_adjusted[0] / points_camera_adjusted[2]) < np.tan(fov_x/2)) & \
             (np.abs(points_camera_adjusted[1] / points_camera_adjusted[2]) < np.tan(fov_y/2)) & \
             (points_camera_adjusted[2] > 0)
    
    points_camera_adjusted = points_camera_adjusted[:, in_fov]
    
    if points_camera_adjusted.shape[1] == 0:
        print("警告：视锥体内没有点")
        return np.array([]), np.array([]), np.array([])
    
    # 执行投影
    points_pixel = np.dot(intrinsic, points_camera_adjusted)
    points_pixel = points_pixel / points_camera_adjusted[2, :]
    
    image_width = 800
    image_height = 600
    
    in_image = (points_pixel[0, :] >= 0) & (points_pixel[0, :] < image_width) & \
               (points_pixel[1, :] >= 0) & (points_pixel[1, :] < image_height)
    
    valid_indices = np.where(in_fov)[0]
    return points_pixel, in_image, valid_indices[in_image]

def process_lidar_points(points_lidar, data, camera_param_key):
    camera_params = data[camera_param_key]
    intrinsic = np.array(camera_params['intrinsic'])
    extrinsic = np.array(camera_params['extrinsic'])
    
    points_pixel, in_image, valid_indices = lidar_to_camera_pixel(points_lidar, data, camera_param_key, intrinsic, extrinsic)
    
    if len(valid_indices) == 0:
        print(f"警告：{camera_param_key} 没有有效点")
        return np.array([])
    
    valid_points = points_lidar[valid_indices[in_image]]
    return valid_points

def create_colored_bev(points_list, camera_names, resolution=0.1, x_range=(-40, 40), y_range=(-40, 40)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_size = int((x_max - x_min) / resolution)
    y_size = int((y_max - y_min) / resolution)
    
    # 创建一个带有背景色的图像
    bev = np.zeros((y_size, x_size, 4), dtype=np.uint8)  # RGBA
    bev[:, :, 3] = 255  # 设置 alpha 通道为不透明

    # 为每个相机分配一个唯一的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(camera_names)))
    color_map = {name: color for name, color in zip(camera_names, colors)}

    # 绘制每个相机的点云
    for points, camera_name in zip(points_list, camera_names):
        if len(points) == 0:
            continue
        x_coords = ((points[:, 0] - x_min) / resolution).astype(int)
        y_coords = ((points[:, 1] - y_min) / resolution).astype(int)
        
        valid_indices = (x_coords >= 0) & (x_coords < x_size) & (y_coords >= 0) & (y_coords < y_size)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]

        # 逆时针旋转90度
        temp = x_coords.copy()
        x_coords = y_coords
        y_coords = temp
        
        bev[y_coords, x_coords, :3] = (color_map[camera_name][:3] * 255).astype(np.uint8)

    return bev

def main():
    fold = "./datasets/kitti/training/image_2"
    yaml_fold = "./datasets/kitti/training/yaml"
    pcd_fold = "./datasets/kitti/testing/pcd"
    label_fold = "./datasets/kitti/testing/label_2"
    calib_fold = "./datasets/kitti/testing/calib"
    
    # 获取所有场景的文件名
    scene_files = sorted([f for f in os.listdir(fold) if f.endswith('_camera0.png')])
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('10_2.mp4', fourcc, 5, (1400, 1000))  # 假设BEV图像大小为1400x1000
    
    for scene_file in tqdm(scene_files, desc="处理场景"):
        file_name = scene_file.split('_')[0]
        
        pcd_file = os.path.join(pcd_fold, f"{file_name}.pcd")
        yaml_file = os.path.join(yaml_fold, f"{file_name}.yaml")
        
        points = read_pcd(pcd_file)
        data = load_yaml(yaml_file)
        
        vehicle_positions = []
        camera_3d_boxes = {}
        camera_params = {}
        all_camera_points = []
        camera_names = []

        for camera_num in range(6):
            camera_param_key = f'camera{camera_num}'
            r3dbbx_key = f'camera_{camera_num}_r3dbbx'

            valid_points = process_lidar_points(points, data, camera_param_key)
            
            if len(valid_points) > 0:
                all_camera_points.append(valid_points)
                camera_names.append(camera_param_key)

            if r3dbbx_key in data:
                camera_3d_boxes[r3dbbx_key] = data[r3dbbx_key]
            
            if camera_param_key in data:
                camera_params[camera_param_key] = data[camera_param_key]

        colored_bev = create_colored_bev(all_camera_points, camera_names)

        for vehicle_id, vehicle_info in data['vehicles'].items():
            location = vehicle_info['location']
            vehicle_positions.append((location[0], location[1]))

        ego_position = data['true_ego_pos'][:2]
        lanes_3d = data['lane3d']

        camera_3d_boxes_pred = {}
        camera_3d_boxes_gt = {}
        for camera_num in range(6):
            camera_param_key = f'camera{camera_num}'
            r3dbbx_key = f'camera_{camera_num}_r3dbbx'

            label_file = os.path.join(label_fold, f"{file_name}_{camera_param_key}.txt")
            calib_file = os.path.join(calib_fold, f"{file_name}_{camera_param_key}.txt") 
            
            camera_3d_boxes_pred[r3dbbx_key] = get_boxs(label_file, calib_file)
            if r3dbbx_key in data:
                camera_3d_boxes_gt[r3dbbx_key] = data[r3dbbx_key]

        save_path = f'temp_bev_images/temp_bev_{file_name}.png'
        
        visualize_bev(data, colored_bev, vehicle_positions, camera_3d_boxes_pred, camera_3d_boxes_gt, lanes_3d, ego_position, camera_params, camera_names, save_path=save_path)
        
        # 读取生成的BEV图像并写入视频
        frame = cv2.imread(save_path)
        video_writer.write(frame)
        
        # 删除临时文件
        # os.remove(save_path)

    # 关闭视频写入器
    video_writer.release()
    print("视频生成完成：bev_visualization.mp4")

if __name__ == "__main__":
    main()