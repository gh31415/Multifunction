import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('analysis results/desert_snow_cosin(-1,1)_circular_10k.csv')
dataset2 = pd.read_csv('analysis results/desert_snow_cosin(-1,1)_circular.csv')
dataset3 = pd.read_csv('analysis results/z_desert_snow_cosin(-1,1)_10k.csv')
dataset4 = pd.read_csv('analysis results/z_desert_snow_cosin(-1,1)_circular.csv')
dataset5 = pd.read_csv('analysis results/desert_snow_cosin(-1,1)_square.csv')
dataset6 = pd.read_csv('analysis results/z_desert_snow_cosin(-1,1)_square.csv')


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


# 计算KL散度
def kl_divergence(hist_1, hist_3):
    # 将hist1和hist3中的0值替换为1e-10，避免出现log(0)的情况
    hist_1[hist_1 == 0] = 1e-10
    hist_3[hist_3 == 0] = 1e-10
    # 计算KL散度
    kl_div = np.sum(hist_1 * np.log(hist_1 / hist_3))
    return kl_div


fig = plt.figure(figsize=(20, 20), dpi=100)
for i in range(1, 5):
    fig.add_subplot(2, 2, i)
    plt.subplot(221)
    values1, bins1, bars1 = plt.hist(dataset5['desert_snow_square'], edgecolor='white', bins=100, color='orange', range=(-1, 1))
    hist1, _ = np.histogram(dataset5['desert_snow_square'], bins=100, density=True)
    plt.ylim(0, 350)
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('square desert&snow cosin(-1,1) 1k')
    count11 = np.count_nonzero((dataset5['desert_snow_square'] >= 0.75) & (dataset5['desert_snow_square'] <= 1))
    count12 = np.count_nonzero((dataset5['desert_snow_square'] >= 0.5) & (dataset5['desert_snow_square'] <= 0.75))
    count13 = np.count_nonzero((dataset5['desert_snow_square'] >= 0.25) & (dataset5['desert_snow_square'] <= 0.5))
    count14 = np.count_nonzero((dataset5['desert_snow_square'] >= 0) & (dataset5['desert_snow_square'] <= 0.25))
    count15 = np.count_nonzero((dataset5['desert_snow_square'] >= -0.25) & (dataset5['desert_snow_square'] <= 0))
    count16 = np.count_nonzero((dataset5['desert_snow_square'] >= -0.5) & (dataset5['desert_snow_square'] <= -0.25))
    count17 = np.count_nonzero((dataset5['desert_snow_square'] >= -0.75) & (dataset5['desert_snow_square'] <= -0.5))
    count18 = np.count_nonzero((dataset5['desert_snow_square'] >= -1) & (dataset5['desert_snow_square'] <= -0.75))
    print('图1' + str((count11, count12, count13, count14, count15, count16, count17, count18)))
    # plt.text(x=-1, y=300, s='cosin-[0.9,1] percentage:' + str(count11 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=275, s='cosin-[-1,0.75] percentage:' + str(count12 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    plt.subplot(222)
    values2, bins2, bars2 = plt.hist(dataset2['desert_snow_circular'], edgecolor='white', bins=100, color='skyblue', range=(-1, 1))
    hist2, _ = np.histogram(dataset2['desert_snow_circular'], bins=100, density=True)
    plt.ylim(0, 350)
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('circular desert&snow cosin(-1,1) 1k')
    count21 = np.count_nonzero((dataset2['desert_snow_circular'] >= 0.75) & (dataset2['desert_snow_circular'] <= 1))
    count22 = np.count_nonzero((dataset2['desert_snow_circular'] >= 0.5) & (dataset2['desert_snow_circular'] <= 0.75))
    count23 = np.count_nonzero((dataset2['desert_snow_circular'] >= 0.25) & (dataset2['desert_snow_circular'] <= 0.5))
    count24 = np.count_nonzero((dataset2['desert_snow_circular'] >= 0) & (dataset2['desert_snow_circular'] <= 0.25))
    count25 = np.count_nonzero((dataset2['desert_snow_circular'] >= -0.25) & (dataset2['desert_snow_circular'] <= 0))
    count26 = np.count_nonzero((dataset2['desert_snow_circular'] >= -0.5) & (dataset2['desert_snow_circular'] <= -0.25))
    count27 = np.count_nonzero((dataset2['desert_snow_circular'] >= -0.75) & (dataset2['desert_snow_circular'] <= -0.5))
    count28 = np.count_nonzero((dataset2['desert_snow_circular'] >= -1) & (dataset2['desert_snow_circular'] <= -0.75))
    print('图2' + str((count21, count22, count23, count24, count25, count26, count27, count28)))
    # plt.text(x=-1, y=300, s='cosin-[0.9,1] percentage:' + str(count21 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=275, s='cosin-[-1,0.75] percentage:' + str(count22 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    plt.subplot(223)
    values3, bins3, bars3 = plt.hist(dataset6['z_desert_snow_square'], edgecolor='white', bins=100, color='orange', range=(-1, 1))
    hist3, _ = np.histogram(dataset6['z_desert_snow_square'], bins=100, density=True)
    plt.ylim(0, 750)
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('z square desert&snow cosin(-1,1) 1k')
    mmd1_3 = mmd(hist1, hist3, sigma=1.0)  # 计算MMD
    kl1_3 = kl_divergence(hist1 / 1000, hist3 / 1000)
    kl3_1 = kl_divergence(hist3 / 1000, hist1 / 1000)
    print((mmd1_3, kl1_3, kl3_1))
    count31 = np.count_nonzero((dataset6['z_desert_snow_square'] >= 0.75) & (dataset6['z_desert_snow_square'] <= 1))
    count32 = np.count_nonzero((dataset6['z_desert_snow_square'] >= 0.5) & (dataset6['z_desert_snow_square'] <= 0.75))
    count33 = np.count_nonzero((dataset6['z_desert_snow_square'] >= 0.25) & (dataset6['z_desert_snow_square'] <= 0.5))
    count34 = np.count_nonzero((dataset6['z_desert_snow_square'] >= 0) & (dataset6['z_desert_snow_square'] <= 0.25))
    count35 = np.count_nonzero((dataset6['z_desert_snow_square'] >= -0.25) & (dataset6['z_desert_snow_square'] <= 0))
    count36 = np.count_nonzero((dataset6['z_desert_snow_square'] >= -0.5) & (dataset6['z_desert_snow_square'] <= -0.25))
    count37 = np.count_nonzero((dataset6['z_desert_snow_square'] >= -0.75) & (dataset6['z_desert_snow_square'] <= -0.5))
    count38 = np.count_nonzero((dataset6['z_desert_snow_square'] >= -1) & (dataset6['z_desert_snow_square'] <= -0.75))
    print('图3' + str((count31, count32, count33, count34, count35, count36, count37, count38)))
    # plt.text(x=-1, y=375, s='MMD:' + str(mmd1_3), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=0, y=380, s='KL1_3:' + str(kl1_3), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=0, y=365, s='KL3_1:' + str(kl3_1), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=300, s='cosin-[0.9,1] percentage:' + str(count31 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=275, s='cosin-[-1,0.75] percentage:' + str(count32 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    plt.subplot(224)
    values4, bins4, bars4 = plt.hist(dataset4['z_desert_snow_circular'], edgecolor='white', bins=100, color='skyblue', range=(-1, 1))
    hist4, _ = np.histogram(dataset4['z_desert_snow_circular'], bins=100, density=True)
    plt.ylim(0, 350)
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('z circular desert&snow cosin(-1,1) 1k')
    mmd2_4 = mmd(hist2, hist4, sigma=1.0)  # 计算MMD
    kl2_4 = kl_divergence(hist2 / 1000, hist4 / 1000)
    kl4_2 = kl_divergence(hist4 / 1000, hist2 / 1000)
    print((mmd2_4, kl2_4, kl4_2))
    count41 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= 0.75) & (dataset4['z_desert_snow_circular'] <= 1))
    count42 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= 0.5) & (dataset4['z_desert_snow_circular'] <= 0.75))
    count43 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= 0.25) & (dataset4['z_desert_snow_circular'] <= 0.5))
    count44 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= 0) & (dataset4['z_desert_snow_circular'] <= 0.25))
    count45 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= -0.25) & (dataset4['z_desert_snow_circular'] <= 0))
    count46 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= -0.5) & (dataset4['z_desert_snow_circular'] <= -0.25))
    count47 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= -0.75) & (dataset4['z_desert_snow_circular'] <= -0.5))
    count48 = np.count_nonzero((dataset4['z_desert_snow_circular'] >= -1) & (dataset4['z_desert_snow_circular'] <= -0.75))
    print('图4' + str((count41, count42, count43, count44, count45, count46, count47, count48)))
    # plt.text(x=-1, y=375, s='MMD:' + str(mmd2_4), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=0, y=380, s='KL2_4:' + str(kl2_4), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=0, y=365, s='KL4_2:' + str(kl4_2), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=300, s='cosin-[0.9,1] percentage:' + str(count41 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.text(x=-1, y=275, s='cosin-[-1,0.75] percentage:' + str(count42 / 1000), fontdict=dict(fontsize=12, color='red', family='monospace', weight='normal'))
    # plt.savefig('E:/FunctionMethod/analysis results/9.png')
plt.show()
