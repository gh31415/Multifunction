# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 定义函数计算RGB频率直方图
# def compute_histogram(image):
#     # 将图像转换为RGB格式
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # 计算每个颜色通道的直方图
#     r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#     g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
#     b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])
#
#     return r_hist, g_hist, b_hist
#
#
# # 创建一个大图像，用于绘制所有直方图
# fig, axs = plt.subplots(3, 1, figsize=(12, 16))
#
# # 遍历10000张图片，并绘制直方图
# for i in range(10000):
#     # 加载图片
#     image = cv2.imread(f"E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/image_{i}.jpg")
#
#     # 计算RGB频率直方图
#     r_hist, g_hist, b_hist = compute_histogram(image)
#
#     # 在大图像上绘制直方图
#     axs[0].plot(r_hist, color='red', alpha=0.2)
#     axs[1].plot(g_hist, color='green', alpha=0.2)
#     axs[2].plot(b_hist, color='blue', alpha=0.2)
#
# # 设置图像标题和标签
# axs[0].set_title('Red Channel Histogram')
# axs[0].set_xlabel('Pixel Value')
# axs[0].set_ylabel('Frequency')
# axs[1].set_title('Green Channel Histogram')
# axs[1].set_xlabel('Pixel Value')
# axs[1].set_ylabel('Frequency')
# axs[2].set_title('Blue Channel Histogram')
# axs[2].set_xlabel('Pixel Value')
# axs[2].set_ylabel('Frequency')
#
# # 调整子图之间的间距
# plt.tight_layout()
# # plt.savefig('E:/FunctionMethod/analysis results/desert_dataset_hist.png')
# # 显示绘制的直方图
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 定义函数计算RGB频率直方图
# def compute_histogram(image):
#     # 将图像转换为RGB格式
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # 计算每个颜色通道的直方图
#     r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#     g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
#     b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])
#
#     hist = r_hist + g_hist + b_hist
#
#     return hist
#
#
# # 创建一个大图像，用于绘制所有直方图
# fig, axs = plt.subplots(1, 1, figsize=(12, 8))
#
# # 遍历10000张图片，并绘制直方图
# for i in range(10000):
#     # 加载图片
#     image = cv2.imread(f"E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/image_{i}.jpg")
#
#     # 计算RGB频率直方图
#     hist = compute_histogram(image)
#
#     # 在大图像上绘制直方图
#     axs.plot(hist, color='red', alpha=0.05)
#
# # 设置图像标题和标签
# axs.set_title('RGB Histogram')
# axs.set_xlabel('Pixel Value')
# axs.set_ylabel('Frequency')
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# plt.savefig('E:/FunctionMethod/analysis results/snow_dataset_total_hist.png')
#
# # 显示绘制的直方图
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 定义函数计算RGB频率直方图和均值
def compute_histogram(image):
    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算每个颜色通道的直方图
    r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    # 计算每个通道的直方图均值
    r_mean = np.mean(r_hist)
    g_mean = np.mean(g_hist)
    b_mean = np.mean(b_hist)

    return r_hist, g_hist, b_hist, r_mean, g_mean, b_mean


# 创建一个大图像，用于绘制所有直方图
fig, axs = plt.subplots(3, 1, figsize=(12, 16))

# 遍历10000张图片，并绘制直方图
for i in range(10000):
    # 加载图片
    image = cv2.imread(f"E:/FunctionMethod/airsim_images/desert_datasets/circular_10k/images/image_{i}.jpg")

    # 计算RGB频率直方图和均值
    r_hist, g_hist, b_hist, r_mean, g_mean, b_mean = compute_histogram(image)

    # 在大图像上绘制直方图
    axs[0].plot(r_hist, color='red', alpha=0.2)
    axs[1].plot(g_hist, color='green', alpha=0.2)
    axs[2].plot(b_hist, color='blue', alpha=0.2)

# 设置图像标题和标签
axs[0].set_title('Red Channel Histogram (Mean: {:.2f})'.format(r_mean))
axs[0].set_xlabel('Pixel Value')
axs[0].set_ylabel('Frequency')
axs[1].set_title('Green Channel Histogram (Mean: {:.2f})'.format(g_mean))
axs[1].set_xlabel('Pixel Value')
axs[1].set_ylabel('Frequency')
axs[2].set_title('Blue Channel Histogram (Mean: {:.2f})'.format(b_mean))
axs[2].set_xlabel('Pixel Value')
axs[2].set_ylabel('Frequency')

# 调整子图之间的间距
plt.tight_layout()
# plt.savefig('E:/FunctionMethod/analysis results/desert_dataset_hist.png')
# 显示绘制的直方图
plt.show()
