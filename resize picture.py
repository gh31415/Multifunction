from PIL import Image
import os

# 设置裁剪后的图像大小
target_size = (120, 120)

# 设置文件夹路径
folder_path = "E:/FunctionMethod/rotate pictures/circular desert"

# 遍历文件夹中所有的png文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 打开源图像文件
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # 裁剪图像
        cropped_img = img.crop((56, 0, 200, 144))

        # 打开原始图片并裁剪
        save_path = os.path.join(folder_path, "cropped_" + filename)
        cropped_img.save(save_path, "PNG")
