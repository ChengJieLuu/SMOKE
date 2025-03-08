import numpy as np
import matplotlib.pyplot as plt

# 世界坐标系下的物体位置，假设已经通过转换得到
object_positions_world = np.array([
    [2.0, 3.0, 1.0],  # 物体1
    [5.0, 7.0, 1.5],  # 物体2
    [1.0, 4.0, 0.5]   # 物体3
])

# 提取X、Y坐标
x_coords = object_positions_world[:, 0]
y_coords = object_positions_world[:, 1]

# 绘制俯视图
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c='r', marker='o')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Top-Down View of Objects in World Coordinate System')
plt.grid(True)

# 保存图片到文件
plt.savefig('top_down_view.png')
