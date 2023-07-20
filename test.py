import numpy as np

data = np.loadtxt('output1.txt')  # 从txt文件加载数据到numpy数组

mean = np.mean(data)  # 计算数据数组的均值

print(mean)  # 输出均值结果
