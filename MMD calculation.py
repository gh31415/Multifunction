import pandas as pd
import numpy as np

# 读数据
dataset1 = pd.read_csv('analysis results/square_circular_cosin(-1,1)_120.csv')
dataset2 = pd.read_csv('analysis results/circular_square_cosin(-1,1)_rectangle.csv')
dataset3 = pd.read_csv('analysis results/z_square_circular_cosin(-1,1)_120.csv')
dataset4 = pd.read_csv('analysis results/z_circular_square_cosin(-1,1)_rectangle.csv')

# 得到数据分布
hist1, _ = np.histogram(dataset1['square_circular'], bins=50, density=True)
hist2, _ = np.histogram(dataset2['circular_square'], bins=50, density=True)
hist3, _ = np.histogram(dataset3['z_square_circular'], bins=50, density=True)
hist4, _ = np.histogram(dataset4['z_circular_square'], bins=50, density=True)


# 计算MMD
def gaussian_kernel(x, y, sigma=1.0):
    gamma = 1.0 / (2 * sigma ** 2)
    pairwise_dists = np.sum(np.square(x), axis=1).reshape(-1, 1) + np.sum(np.square(y), axis=1) - 2 * np.dot(x, y.T)
    return np.exp(-gamma * pairwise_dists)


def mmd(hist_1, hist_2, sigma=1.0):
    K11 = gaussian_kernel(hist_1.reshape((-1, 1)), hist_1.reshape((-1, 1)), sigma)
    K22 = gaussian_kernel(hist_2.reshape((-1, 1)), hist_2.reshape((-1, 1)), sigma)
    K12 = gaussian_kernel(hist_1.reshape((-1, 1)), hist_2.reshape((-1, 1)), sigma)
    MMD = np.sqrt(K11.mean() + K22.mean() - 2 * K12.mean())
    return MMD


mmd1_3 = mmd(hist1, hist3, sigma=1.0)
mmd2_4 = mmd(hist2, hist4, sigma=1.0)
print('mmd1_3:', mmd1_3)
print('mmd2_4:', mmd2_4)
