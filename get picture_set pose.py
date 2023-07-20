# import airsim
# import csv
# import os
# import numpy as np
#
# # 连接到 AirSim
# client = airsim.VehicleClient()
#
# # 设置无人机起点位置
# start_point = airsim.Pose(airsim.Vector3r(0, 0, -3), airsim.to_quaternion(0, 0, 0))
# client.simSetVehiclePose(start_point, True)
#
# # 加载无人机位姿CSV文件，格式为x,y,z,yaw,pitch,roll
# path = 'E:/FunctionMethod/airsim_images'
# with open('airsim_images/poses.csv', newline='') as csvfile:
#     csv_reader = csv.reader(csvfile, delimiter=',')
#     # 跳过第一行
#     next(csv_reader)
#     for row in csv_reader:
#
#         x, y, z, yaw, pitch, roll = map(float, row)
#
#         # 控制无人机飞行到指定位置
#         target_point = airsim.Vector3r(x, y, z)
#         yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
#         client.simSetVehiclePose(airsim.Pose(target_point, airsim.to_quaternion(pitch, roll, yaw)), True)
#
#         # 设置相机位置和方向
#         camera_pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
#         client.simSetCameraPose("0", camera_pose)
#
#         # 获取图像，并保存到本地文件（RGB格式）
#         responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
#         for i, response in enumerate(responses):
#             filename = path + "airsim_image_" + str(i) + '.png'
#             img_raw = responses[0]
#
#             # 将字节流转换为PIL的Image对象
#             img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
#             img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
#             img_rgb1 = np.flipud(img_rgb)
#             airsim.write_png(os.path.normpath(filename + '.png'), img_rgb1)

import numpy as np
import pandas as pd
import airsim
import time

# 读取位姿信息
df = pd.read_csv('../poses.csv')
x, y, z, yaw, pitch, roll = df.iloc[1]
camera_list = ["0", "1", "2", "3", "4"]

# 连接到 AirSim 模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 设置无人机位置和姿态
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
client.simSetVehiclePose(pose, True)
client.hoverAsync().join()
time.sleep(2)

# # 获取无人机视角的单张图像
# response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
# img1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
# img_rgb = img1d.reshape(response[0].height, response[0].width, 3)
for j, camera_name in enumerate(camera_list):
    responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]
    # 将字节流转换为PIL的Image对象
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)
    img_filename = "{}.png".format(j)
    airsim.write_png(img_filename, img_rgb1)
# # 保存图像
# filename = 'airsim_image.png'
# airsim.write_png(filename, img_rgb)
