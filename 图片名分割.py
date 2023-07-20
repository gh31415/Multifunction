import os
import pandas as pd

# 遍历图像数据集所在的文件夹
folder_path = 'E:/FunctionMethod/airsim_images/desert_datasets/circular_10k9'
file_names = os.listdir(folder_path)

# 将文件名以下划线为分隔符分开，并存储在一个列表中
file_info = []
for file_name in file_names:
    file_name_split = file_name.split('_')
    file_info.append(file_name_split)

# 将列表转换为DataFrame，并保存为csv文件
df = pd.DataFrame(file_info)
df.to_csv('E:/FunctionMethod/airsim_images/desert_datasets/circular_10k9/image_dataset.csv', index=False)
