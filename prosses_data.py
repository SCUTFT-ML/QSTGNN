import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# coordinates = pd.read_csv('data/PM25_36/vales36.csv')
# data = coordinates.values

# # 创建一个空的距离矩阵
# num_points = data.shape[1]
# distance_matrix = np.zeros((num_points, num_points))

# # 填充距离矩阵
# for i in range(num_points):
#     for j in range(i + 1, num_points):
#         print(i,j)
#         x = coordinates.values[:,i].reshape(1, -1)
#         y = coordinates.values[:,j].reshape(1, -1)
#         # 计算第i个点和第j个点之间的距离
#         distance_matrix[i, j], path = fastdtw(x, y, dist=euclidean)
#         distance_matrix[j, i] = distance_matrix[i, j]

# np.save('adj_dtw_dis.npy', distance_matrix)

# # 打印距离矩阵
# print(distance_matrix)

# 下面是用插值法补全数据(因为插补方法无法对所有的缺失值进行不全，头尾的缺失值不行，然后还用bfill方法对首尾补全)
# 读取CSV文件
coordinates = pd.read_csv('data/PM25_437/vales437.csv', header=None)
coordinates.interpolate(method='linear', inplace=True)
coordinates = coordinates.fillna(method='bfill')
data = coordinates.values

# 创建一个空的距离矩阵
num_points = data.shape[1]
distance_matrix = np.zeros((num_points, num_points))

# 填充距离矩阵
for i in range(num_points):
    for j in range(i + 1, num_points):
        print(i,j)
        x = coordinates.values[:,i].reshape(1, -1)
        y = coordinates.values[:,j].reshape(1, -1)
        # 计算第i个点和第j个点之间的距离
        distance_matrix[i, j], path = fastdtw(x, y, dist=euclidean)
        distance_matrix[j, i] = distance_matrix[i, j]

np.save('adj_dtw_dis.npy', distance_matrix)

# print(df.isnull().sum())
# print(np.isnan(df.values).any())
# print(np.isinf(df.values).any())