from PIL import Image
import os

# 设置文件夹路径
folder_path = "E:/FunctionMethod/resized pictures/circular desert 144"

# 遍历文件夹中所有的png文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 打开源图像文件
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # 裁剪图像
        resized_img = img.resize((120, 120), resample=Image.BICUBIC)

        # 打开原始图片并裁剪
        save_path = os.path.join(folder_path, "resized_" + filename)
        resized_img.save(save_path, "PNG")
