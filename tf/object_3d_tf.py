import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 示例数据
left_points = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10]
])

right_points = np.array([
    [1.5, 2.5, 3.5],
    [2.5, 3.5, 4.5],
    [3.5, 4.5, 5.5],
    [4.5, 5.5, 6.5],
    [5.5, 6.5, 7.5],
    [6.5, 7.5, 8.5],
    [7.5, 8.5, 9.5],
    [8.5, 9.5, 10.5]
])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制左摄像头结果
ax.scatter(left_points[:, 0], left_points[:, 1], left_points[:, 2], c='r', marker='o', label='Left Camera')

# 绘制右摄像头结果
ax.scatter(right_points[:, 0], right_points[:, 1], right_points[:, 2], c='b', marker='^', label='Right Camera')

# 绘制立方体框
def draw_cube(points, ax, color='r'):
    # 12条边的两个端点
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 下底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 上底面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
    ]
    
    for edge in edges:
        ax.plot3D(*zip(points[edge[0]], points[edge[1]]), color=color)

# 绘制左摄像头的3D框
draw_cube(left_points, ax, color='r')

# 绘制右摄像头的3D框
draw_cube(right_points, ax, color='b')

# 添加坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置图例
ax.legend()

# 显示图形
plt.show()



# 保存图片到文件
plt.savefig('3d_view.png')
