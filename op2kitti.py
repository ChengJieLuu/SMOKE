import yaml
import numpy as np
import os
import math

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_kitti_calib(data, output_file, camera_index):
    camera_data = data[f'camera{camera_index}']
    intrinsic = np.array(camera_data['intrinsic'])
    extrinsic = np.array(camera_data['extrinsic'])
    
    with open(output_file, 'w') as f:
        for i in range(6):
            P = np.zeros((3, 4))
            P[:3, :3] = intrinsic
            # P = intrinsic @ extrinsic[:3, :]
            f.write(f"P{i}: {' '.join(map(str, P.flatten()))}\n")
        
        R0_rect = np.eye(3)
        f.write(f"R0_rect: {' '.join(map(str, R0_rect.flatten()))}\n")
        
        Tr_velo_to_cam = np.linalg.inv(extrinsic)[:3, :]
        f.write(f"Tr_velo_to_cam: {' '.join(map(str, Tr_velo_to_cam.flatten()))}\n")
        
        # 假设 Tr_imu_to_velo 为单位矩阵
        Tr_imu_to_velo = np.eye(4)[:3, :]
        f.write(f"Tr_imu_to_velo: {' '.join(map(str, Tr_imu_to_velo.flatten()))}\n")

def inverse_transform(camera_bbox, camera_k_matrix):
    camera_bbox = camera_bbox[:8]
    camera_bbox = np.array(camera_bbox)
    new_x = camera_bbox[:, 0].reshape(8, 1)
    new_y = camera_bbox[:, 1].reshape(8, 1)
    new_z = camera_bbox[:, 2].reshape(8, 1)

    bbox = np.concatenate([new_x * new_z, new_y * new_z, new_z], axis=1)
    bbox = np.transpose(bbox)  # Shape: (3, 8)

    cords_y_minus_z_x = np.linalg.solve(camera_k_matrix, bbox)

    cords_x_y_z = np.zeros((3, 8))
    cords_x_y_z[0, :] = cords_y_minus_z_x[2, :]
    cords_x_y_z[1, :] = cords_y_minus_z_x[0, :]
    cords_x_y_z[2, :] = -cords_y_minus_z_x[1, :]

    return cords_x_y_z



def compute_rotation_y(camera_coords, dimensions):
    """
    通过比较旋转前后的边界框点来计算rotation_y
    
    参数:
    camera_coords: 相机坐标系中的8个顶点坐标 (3, 8)
    dimensions: 物体的尺寸 [长, 宽, 高]
    
    返回:
    rotation_y: 物体绕y轴的旋转角度
    """
    l, w, h = dimensions
    
    # 构建未旋转的边界框点
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # 计算边界框中心
    box_center = np.mean(camera_coords[:,:4], axis=1)
    
    # 将相机坐标系中的点平移到原点
    centered_coords = camera_coords - box_center.reshape(3, 1)
    
    # 只考虑xz平面的旋转（绕y轴）
    centered_coords_xz = centered_coords[[0, 2], :]
    corners_3d_xz = corners_3d[[0, 2], :]
    
    # 使用SVD求解最优旋转矩阵
    H = np.dot(corners_3d_xz, centered_coords_xz.T)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 从旋转矩阵中提取rotation_y
    rotation_y = np.arctan2(R[0, 1], R[0, 0])
    
    return rotation_y



def generate_kitti_label(data, output_file, camera_index):
    camera_data = data[f'camera{camera_index}']
    camera_location = camera_data['cords'][:3]
    camera_angle = camera_data['cords'][3:]
    intrinsic = np.array(camera_data['intrinsic'])
    extrinsic = np.array(camera_data['extrinsic'])
    
    bbx_key = f'camera_{camera_index}_bbx'
    r3dbbx_key = f'camera_{camera_index}_r3dbbx'
    
    if bbx_key not in data or r3dbbx_key not in data:
        print(f"No bounding box data for camera{camera_index}")
        return
    
    with open(output_file, 'w') as f:
        for bbx, r3dbbx in zip(data[bbx_key], data[r3dbbx_key]):
            # 获取2D边界框
            x_min, y_min = bbx[0]
            x_max, y_max = bbx[1]
            
            # 使用inverse_transform将3D边界框转换回相机坐标系
            camera_coords = inverse_transform(r3dbbx[:8], intrinsic)
            
            # 计算3D边界框中心坐标
            box_center = np.mean(camera_coords[:,:4], axis=1)
            
            # 获取车辆信息
            vehicle = data['vehicles'][r3dbbx[8]]
            dimensions = np.array(vehicle['extent']) * 2
            
            # 计算rotation_y
            rotation_y = compute_rotation_y(camera_coords, dimensions)
            
            # 计算theta（目标中心到相机的射线与x轴的夹角）
            theta = np.arctan2(box_center[2], box_center[0])
            
            # 计算alpha
            alpha = rotation_y - theta
            
            # 确保alpha在[-pi, pi]范围内
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
            
            # 使用camera_coords中的坐标
            x_kitti = box_center[1]
            y_kitti = -box_center[2]
            z_kitti = box_center[0]
            
            # 格式化KITTI标签行
            label_line = (
                f"Car "                           # 类型：统一为 "Car"
                f"0 "                             # 截断程度：默认为0
                f"0 "                             # 遮挡程度：默认为0
                f"{alpha:.6f} "                   # 观测角：alpha
                f"{x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "  # 2D边界框
                f"{dimensions[2]:.2f} "           # 高度
                f"{dimensions[1]:.2f} "           # 宽度
                f"{dimensions[0]:.2f} "           # 长度
                f"{x_kitti:.2f} {y_kitti:.2f} {z_kitti:.2f} "  # 3D边界框中心坐标
                f"{rotation_y:.6f}\n"             # 目标朝向角：rotation_y
            )
            
            f.write(label_line)

def main():
    calib_folder = "datasets/kitti/training/calib"
    label_folder = "datasets/kitti/training/label_2"
    os.makedirs(calib_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    fold = "datasets/kitti/training/yaml"
    for file in os.listdir(fold):
        yaml_file = os.path.join(fold, file)
        base_name = os.path.splitext(os.path.basename(yaml_file))[0]
    
        data = load_yaml(yaml_file)
        
        for i in range(6):  # 处理6个摄像头
            calib_file = os.path.join(calib_folder, f"{base_name}_camera{i}.txt")
            label_file = os.path.join(label_folder, f"{base_name}_camera{i}.txt")
            
            save_kitti_calib(data, calib_file, i)
            generate_kitti_label(data, label_file, i)
        
        print(f"Calibration and label data for camera{i} saved.")

if __name__ == "__main__":
    main()