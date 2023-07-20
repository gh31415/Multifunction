from PIL import Image
import os

# 设置文件夹路径
folder_path = "E:/FunctionMethod/airsim_images/snow_datasets/circular"

# 遍历文件夹中所有的png文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 打开源图像文件
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # 旋转图像
        rotated_img = img.rotate(180)

        # 保存旋转后的图像文件
        save_path = os.path.join(folder_path, "rotated_" + filename)
        rotated_img.save(save_path, "PNG")
