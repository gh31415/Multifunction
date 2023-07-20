# import os
#
#
# class BatchRename:
#     """
#     批量重命名文件夹中的图片文件
#     """
#
#     def __init__(self):
#         self.path = 'E:/FunctionMethod/airsim_images/desert_datasets/circular_10k/images'  # 表示需要命名处理的文件夹
#
#     def rename(self):
#         filelist = os.listdir(self.path)  # 获取文件路径
#         total_num = len(filelist)  # 获取文件长度（个数）
#         i = 0  # 表示文件的命名是从1开始的
#         for item in filelist:
#             if item.endswith('.png'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其
#                 # 他格式，后面的转换格式就可以调整为自己需要的格式即可）
#                 src = os.path.join(os.path.abspath(self.path), item)
#                 # dst = os.path.join(os.path.abspath(self.path), '' + str(i) + '.jpg')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
#                 dst = os.path.join(os.path.abspath(self.path), '0' + format(str(i), '0>3s') + '.png')
#                 # 这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
#                 os.rename(src, dst)
#                 print('converting %s to %s ...' % (src, dst))
#                 i = i + 1
#         print('total %d to rename & converted %d pngs' % (total_num, i))
#
#
# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()

import os

# 指定图片所在的目录路径
image_directory = "E:/FunctionMethod/airsim_images/snow_datasets/circular_10k/images/"

# 遍历目录中的所有文件
for i, filename in enumerate(os.listdir(image_directory)):
    # 构建新的文件名
    new_filename = f"image_{i}.jpg"

    # 构建旧文件的完整路径和新文件的完整路径
    old_filepath = os.path.join(image_directory, filename)
    new_filepath = os.path.join(image_directory, new_filename)

    # 重命名文件
    os.rename(old_filepath, new_filepath)
