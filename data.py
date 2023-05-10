import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def cal_model(k, m):
    """
    Calculate modal shapes and frequencies given stiffness and mass vectors.

    Args:
        k (list or np.array): Stiffness vector (n elements)
        m (list or np.array): Mass vector (n elements)

    Returns:
        tuple: Normalized modal shapes (n x n array) and frequencies (n elements)
    """
    # 将质量和刚度向量转化为质量矩阵和刚度矩阵
    n = len(k)  # 自由度
    k11 = k[1:]
    K = np.diag(k + np.concatenate((k11, [0]))) - np.diag(k11, -1) - np.diag(k11, 1)  # K-Stiffness matrix
    M = np.diag(m)

    # 求解:频率（Hz）frequency, n个
    # 求解振型矩阵d
    d, z = np.linalg.eig(np.linalg.inv(M).dot(K))  # z, d 是 KM 的特征值和特征向量，z 是振型，d 是频率平方
    idx = np.argsort(d)
    d = np.sort(d)
    d = np.diag(d)
    z = z[:, idx]

    # z, d = np.linalg.eig(a, b)
    #
    # # 对广义特征值按大小排序
    # idx = np.argsort(np.abs(np.diag(d)))[::-1]
    # z_sorted = z[:, idx]
    # d_sorted = np.diag(np.diag(d)[idx])

    frequency = np.sqrt(d).real  # frequency 是 n 维的频率向量，单位 Hz
    frequency = np.diag(frequency) / 2.0 / np.pi
    # 振型归一化处理
    z = z / np.sqrt(np.sum(z ** 2, axis=0))

    # 振型第一行为正
    for iii in range(n):
        if z[0, iii] < 0:
            z[:, iii] = z[:, iii] * -1

    return z, frequency


def generate_data(length=6, human=3):
    k = np.random.uniform(1e2, 1e3, size=6)
    k = np.sort(k)[::-1]
    m = np.random.uniform(1e-2, 1e-1, size=6)
    m = np.sort(m)[::-1]
    z, d = cal_model(k, m)
    zd = np.concatenate((z[:, :human], d[:human].reshape(-1, 1).T), axis=0)
    km = np.concatenate((k.reshape(-1, 1).T,m.reshape(-1, 1).T), axis=0)
    return km,zd


def generate_batch(length=6, nums=1000):
    data = []
    for i in range(nums):
        km, zd = generate_data(length)
        # scaler_zd = StandardScaler()
        # zd = scaler_zd.fit_transform(zd)
        #
        # # 对 km 进行标准化
        # scaler_km = StandardScaler()
        # km = scaler_km.fit_transform(km)
        data.append((km, zd))

    return data
if __name__ == '__main__':

    km,zd= generate_data(6, 1)
    print(km)
    print(km.shape)
    print(zd)
    print(zd.shape)


'''

这是标签
tensor([200, 200, 100, 200, 100, 200])
tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000])



这是数据
0.10077136  0.303531    0.51489327 
0.19697338  0.49112347  0.57650377 
0.37151449  0.49112347 -0.31531754 
0.44193927  0.303531   -0.48364055 
0.54271062 -0.303531    0.03125272 
0.56848788 -0.49112347  0.26118624 
4.25880473 12.36067977 18.76532081 

用torch写一个简单的网络来训练

|

-----
-   -
-----

[200, 400, 10, 200, 10, 200]
[0.5000, 0.5000, 0.00, 0.5000, 0.5000, 0.5000]

[200, 200, 100, 200, 100, 200]
[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]


[200, 400, 10, 200, 10, 200]
[200, 200, 100, 200, 100, 200]
mse  ->  100 + 200 = 300 

mse -> 30 + 100 = 130 

mse -> 0.1 + 0.3 = 0.4



'''