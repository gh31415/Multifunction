import numpy as np
from scipy.stats import wasserstein_distance
from PIL import Image

dataset1 = []
for i in range(10000):
    if i < 10:
        img = Image.open('E:/FunctionMethod/airsim_images/desert_datasets/circular_10k/images/000' + str(i) + '.png')
        img_arr = np.array(img)
        dataset1.append(img_arr.ravel())
    elif 10 <= i < 100:
        img = Image.open('E:/FunctionMethod/airsim_images/desert_datasets/circular_10k/images/00' + str(i) + '.png')
        img_arr = np.array(img)
        dataset1.append(img_arr.ravel())
    else:
        img = Image.open('E:/FunctionMethod/airsim_images/desert_datasets/circular_10k/images/0' + str(i) + '.png')
        img_arr = np.array(img)
        dataset1.append(img_arr.ravel())

dataset2 = []
for i in range(10000):
    if i < 10:
        img = Image.open('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/000' + str(i) + '.png')
        img_arr = np.array(img)
        dataset2.append(img_arr.ravel())
    elif 10 <= i < 100:
        img = Image.open('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/00' + str(i) + '.png')
        img_arr = np.array(img)
        dataset2.append(img_arr.ravel())
    else:
        img = Image.open('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/0' + str(i) + '.png')
        img_arr = np.array(img)
        dataset2.append(img_arr.ravel())

emd_distance = 0
for i in range(10000):
    emd_distance = emd_distance + wasserstein_distance(dataset1[i], dataset2[i])
print(emd_distance/10000)
