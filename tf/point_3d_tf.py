from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 世界坐标系下的物体位置
object_positions_world = np.array([
    [2.0, 3.0, 1.0],  # 物体1
    [5.0, 7.0, 1.5],  # 物体2
    [1.0, 4.0, 0.5]   # 物体3
])

# 提取X、Y、Z坐标
x_coords = object_positions_world[:, 0]
y_coords = object_positions_world[:, 1]
z_coords = object_positions_world[:, 2]

# 绘制3D图形
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D View of Objects in World Coordinate System')

# 保存图片到文件
plt.savefig('3d_view.png')
