import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

dataset1 = pd.read_csv('analysis results/desert_circular_itself_cosin(-1,1).csv')
dataset2 = pd.read_csv('analysis results/snow_circular_itself_cosin(-1,1).csv')
dataset3 = pd.read_csv('analysis results/z_desert_circular_itself_cosin.csv')
dataset4 = pd.read_csv('analysis results/z_snow_circular_itself_cosin.csv')


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


# # 计算KL散度
# def kl_divergence(hist_1, hist_2):
#     # 将hist1和hist3中的0值替换为1e-10，避免出现log(0)的情况
#     hist_1[hist_1 == 0] = 1e-10
#     hist_2[hist_2 == 0] = 1e-10
#     # 计算KL散度
#     kl_div = np.sum(hist_1 * np.log(hist_1 / hist_2))
#     return kl_div


fig = plt.figure(figsize=(20, 20), dpi=100)
for i in range(1, 5):
    fig.add_subplot(2, 2, i)
    plt.subplot(221)
    values1, bins1, bars1 = plt.hist(dataset1['desert_circular_itself'], edgecolor='white', bins=100, color='orange', range=(-1, 1))
    hist1, _ = np.histogram(dataset1['desert_circular_itself'], bins=100, density=True)
    hist1 = np.where(hist1 == 0, 1e-10, hist1)
    plt.ylim(0, 1350)
    plt.xlabel("cosin simi")
    plt.ylabel("Frequency")
    plt.title('desert circular itself')
    plt.subplot(222)
    values2, bins2, bars2 = plt.hist(dataset2['snow_circular_itself'], edgecolor='white', bins=100, color='skyblue', range=(-1, 1))
    hist2, _ = np.histogram(dataset2['snow_circular_itself'], bins=100, density=True)
    hist2 = np.where(hist2 == 0, 1e-10, hist2)
    plt.ylim(0, 1350)
    plt.xlabel("cosin simi")
    plt.ylabel("Frequency")
    plt.title('snow circular itself')
    MMD1 = mmd(hist1, hist2, sigma=1.0)
    KL12 = stats.entropy(hist1, hist2)
    KL21 = stats.entropy(hist2, hist1)
    print('MMD1:', MMD1)
    print('KL12:', KL12)
    print('KL21:', KL21)
    plt.subplot(223)
    values3, bins3, bars3 = plt.hist(dataset3['z_desert_circular_itself'], edgecolor='white', bins=100, color='orange', range=(-1, 1))
    hist3, _ = np.histogram(dataset3['z_desert_circular_itself'], bins=100, density=True)
    hist3 = np.where(hist3 == 0, 1e-10, hist3)
    plt.ylim(0, 1350)
    plt.xlabel("cosin simi")
    plt.ylabel("Frequency")
    plt.title('z desert circular itself')
    plt.subplot(224)
    values4, bins4, bars4 = plt.hist(dataset4['z_snow_circular_itself'], edgecolor='white', bins=100, color='skyblue', range=(-1, 1))
    hist4, _ = np.histogram(dataset4['z_snow_circular_itself'], bins=100, density=True)
    hist4 = np.where(hist4 == 0, 1e-10, hist4)
    plt.ylim(0, 1350)
    plt.xlabel("cosin simi")
    plt.ylabel("Frequency")
    plt.title('z snow circular itself')
    MMD2 = mmd(hist3, hist4, sigma=1.0)
    KL34 = stats.entropy(hist3, hist4)
    KL43 = stats.entropy(hist4, hist3)
    print('MMD2:', MMD2)
    print('KL34:', KL34)
    print('KL43:', KL43)
    # plt.savefig('E:/FunctionMethod/analysis results/11.png')
plt.show()
