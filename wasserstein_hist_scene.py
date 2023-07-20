import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('analysis results/desert_snow_wasserstein.csv')
dataset2 = pd.read_csv('analysis results/z_desert_snow_wasserstein.csv')

fig = plt.figure(figsize=(10, 12), dpi=100)
for i in range(1, 3):
    fig.add_subplot(2, 2, i)
    plt.subplot(211)
    values1, bins1, bars1 = plt.hist(dataset1['desert_snow_wasserstein'], edgecolor='white', bins=100, color='orange', range=(0, 140))
    hist1, _ = np.histogram(dataset1['desert_snow_wasserstein'], bins=100, density=True)
    plt.xlabel("emd_distance")
    plt.ylabel("Frequency")
    plt.title('desert snow emd_distance')
    plt.subplot(212)
    values2, bins2, bars2 = plt.hist(dataset2['z_desert_snow_wasserstein'], edgecolor='white', bins=100, color='skyblue', range=(0, 20))
    hist2, _ = np.histogram(dataset2['z_desert_snow_wasserstein'], bins=100, density=True)
    plt.xlabel("emd_distance")
    plt.ylabel("Frequency")
    plt.title('z desert snow emd_distance')
    plt.savefig('analysis results/12.png')
plt.show()
