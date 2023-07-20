from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def process_image(file_path):
    # 打开并加载图片
    im = Image.open(file_path)
    pixels = np.array(im)

    # 将像素矩阵变形为二维数组
    h, w, d = pixels.shape
    pixels_2d = pixels.reshape(h * w, d)

    # 使用 K-means 算法将像素矩阵分成 3 个颜色簇
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels_2d)
    labels = kmeans.predict(pixels_2d)

    # 计算每个颜色簇的像素数
    label_counts = np.bincount(labels)

    # 找到像素数量最多的颜色簇
    background_color_index = np.argmax(label_counts)

    # 将背景中的主要颜色置为白色
    pixels_2d[labels == background_color_index] = [255, 255, 255]
    pixels = pixels_2d.reshape(h, w, d)

    # 保存修改后的图像
    im_out = Image.fromarray(pixels)
    im_out.save(file_path)


# 处理1000张图片
for i in range(9000):
    if i < 10:
        in_path = 'E:/FunctionMethod/airsim_images/snow_datasets/circular_10k1/000' + str(i) + '.png'
        process_image(in_path)
    elif 10 <= i < 100:
        in_path = 'E:/FunctionMethod/airsim_images/snow_datasets/circular_10k1/00' + str(i) + '.png'
        process_image(in_path)
    else:
        in_path = 'E:/FunctionMethod/airsim_images/snow_datasets/circular_10k1/0' + str(i) + '.png'
        process_image(in_path)
