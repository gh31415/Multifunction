import cv2
import matplotlib.pyplot as plt
import torchvision as torchvision
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# print('----------------------单张图片余弦相似度--------------------')
# img1 = cv2.imread('E:/FunctionMethod/resized pictures/desert 144/0000.png')
# img2 = cv2.imread('E:/FunctionMethod/resized pictures/snow 144/0000.png')
# img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img3_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# print(img1_rgb)
# img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
# img3_rgb_flat = np.array(img3_rgb.ravel()).astype(int)
# print(img1_rgb_flat)
# print(img3_rgb_flat)
# print('------------------------------------------')
# img1_rgb_flat1 = img1_rgb_flat / (np.linalg.norm(img1_rgb_flat))
# img3_rgb_flat1 = img3_rgb_flat / (np.linalg.norm(img3_rgb_flat))
# print(img1_rgb_flat1)
# print(img3_rgb_flat1)
# cos_simi = np.dot(img1_rgb_flat, img3_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img3_rgb_flat))
# cos_simi1 = np.sum(img1_rgb_flat1 * img3_rgb_flat1) / (np.linalg.norm(img1_rgb_flat1) * np.linalg.norm(img3_rgb_flat1))
# cos_simi2 = cosine_similarity(img1_rgb_flat1.reshape(1, -1), img3_rgb_flat1.reshape(1, -1))
# print('余弦相似度', cos_simi)
# print('余弦相似度', cos_simi1)
# print('余弦相似度', cos_simi2[0][0])


# print('----------------------相同序号余弦相似度--------------------')
# f = open('analysis results/circular_square_cosin(-1,1)_circular_8k.txt', 'w')
# for i in range(0, 8600):
#     if i < 10:
#         img1 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/circular_8k/images/000' + str(i) + '.png')
#         img2 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/square_8k/images/000' + str(i) + '.png')
#         img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#         img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#         img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#         img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
#         img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
#         cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#         f.write(str(cos_simi) + '\n')
#     elif 10 <= i < 100:
#         img1 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/circular_8k/images/00' + str(i) + '.png')
#         img2 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/square_8k/images/00' + str(i) + '.png')
#         img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#         img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#         img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#         img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
#         img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
#         cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#         f.write(str(cos_simi) + '\n')
#     else:
#         img1 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/circular_8k/images/0' + str(i) + '.png')
#         img2 = cv2.imread('E:/FunctionMethod/airsim_images/desert_datasets/square_8k/images/0' + str(i) + '.png')
#         img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#         img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#         img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#         img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
#         img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
#         cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#         f.write(str(cos_simi) + '\n')


# print('----------------------不同序号两两之间余弦相似度--------------------')
# f = open('cosin_similarity_all(-1,1).txt', 'w')
# for i in range(0, 1000):
#     for j in range(0, 1000):
#         if i < 10:
#             # img1 = cv2.imread('E:/Datasets/snow_data_1k/images/000' + str(i) + '.png')
#             img1 = cv2.imread('E:/Datasets/desert_data_1k/images/000' + str(i) + '.png')
#             if j < 10:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/000'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             elif 10 <= j < 100:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/00'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             else:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/0'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#         elif 10 <= i < 100:
#             # img1 = cv2.imread('E:/Datasets/snow_data_1k/images/00' + str(i) + '.png')
#             img1 = cv2.imread('E:/Datasets/desert_data_1k/images/00' + str(i) + '.png')
#             if j < 10:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/000'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             elif 10 <= j < 100:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/00'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             else:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/0'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#         else:
#             # img1 = cv2.imread('E:/Datasets/snow_data_1k/images/0' + str(i) + '.png')
#             img1 = cv2.imread('E:/Datasets/desert_data_1k/images/0' + str(i) + '.png')
#             if j < 10:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/000'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             elif 10 <= j < 100:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/00'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
#             else:
#                 img2 = cv2.imread('E:/AirSim-Drone-Racing-VAE-Imitation/airsim_datasets/soccer_close_1k/images/0'+str(j)+'.png')
#                 img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#                 img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#                 img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
#                 img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
#                 img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
#                 img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
#                 cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
#                 f.write('第' + str(i) + '张与第' + str(j) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')


print('----------------------同一数据集下余弦相似度--------------------')
f = open('analysis results/snow_circular_itself_cosin(-1,1).txt', 'w')
img1 = cv2.imread('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/0000.png')
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
img1_rgb_flat = img1_rgb_flat / 255.0 * 2.0 - 1.0
for i in range(0, 10000):
    if i < 10:
        img2 = cv2.imread('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/000' + str(i) + '.png')
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write(str(cos_simi)+'\n')
    elif 10 <= i < 100:
        img2 = cv2.imread('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/00' + str(i) + '.png')
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write(str(cos_simi)+'\n')
    else:
        img2 = cv2.imread('E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/0' + str(i) + '.png')
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img2_rgb_flat = img2_rgb_flat / 255.0 * 2.0 - 1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write(str(cos_simi)+'\n')



