# import numpy as np
# import matplotlib.pyplot as plt
#
# # k 值列表
# k_values = []
# for i in range(4):
#     k = input('请输入第{}个参数：'.format(i+1))
#     k_values.append(int(k))
#
# # d 取值范围
# d = np.linspace(0, 6000, 1000)
#
# # 绘制曲线
# for k in k_values:
#     # 计算概率
#     probability = np.exp(-d ** 2 / (2 * k ** 2))
#
#     # 设置曲线颜色并绘制
#     plt.plot(d, probability, label='Ds = {}'.format(k))
#
# # 图形设置
# plt.title('Probability Function')
# plt.xlabel('Distance (m)')
# plt.ylabel('Probability (P)')
# plt.xlim(0, 6000)
# plt.ylim(0, 1)
# plt.grid(True)
# plt.legend()
# plt.show()


