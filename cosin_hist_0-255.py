import pandas as pd
import matplotlib.pyplot as plt

reviews1 = pd.read_csv('desert_cosin_similarity0-255.csv')
reviews2 = pd.read_csv('desert_cosin_same_0-255.csv')

fig = plt.figure(figsize=(8, 10), dpi=100)
for i in range(1, 3):
    fig.add_subplot(2, 1, i)
    plt.subplot(211)
    values1, bins1, bars1 = plt.hist(reviews1['0-255'], edgecolor='white', bins=50, color='orange', range=(0.4, 1))
    plt.ylim(0, 250)
    plt.xlabel("formal_0-255")
    plt.ylabel("Number")
    plt.bar_label(bars1, fontsize=8, color='red')
    plt.subplot(212)
    values2, bins2, bars2 = plt.hist(reviews2['same_0-255'], edgecolor='white', bins=50, color='orange', range=(0.4, 1))
    plt.ylim(0, 250)
    plt.xlabel("same_scene_0-255")
    plt.ylabel("Number")
    plt.bar_label(bars2, fontsize=8, color='red')
plt.show()
