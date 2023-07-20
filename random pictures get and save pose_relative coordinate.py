import airsim
import numpy as np
import math
import csv
import os

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取对API的控制权
client.enableApiControl(True)

# 设置保存目录，确保目录存在
save_dir = "E:/FunctionMethod/airsim_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义生成随机点的最大和最小范围
x_min, x_max = -100, 100
y_min, y_max = -100, 100
z_min, z_max = -50, -30

num_samples = 1000  # 要生成的采样数量
rows = []  # 要保存到CSV中的数据

for i in range(num_samples):
    # 获取无人机当前状态
    state = client.getMultirotorState()
    x0 = state.kinematics_estimated.position.x_val
    y0 = state.kinematics_estimated.position.y_val
    z0 = state.kinematics_estimated.position.z_val
    yaw = math.degrees(airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2])

    # 生成随机偏移量
    dx = np.random.uniform(x_min, x_max)
    dy = np.random.uniform(y_min, y_max)
    dz = np.random.uniform(z_min, z_max)

    # 将偏移量转换为相对于无人机当前位置的坐标
    d_x, d_y = dx * math.cos(math.radians(yaw)) - dy * math.sin(math.radians(yaw)), dx * math.sin(math.radians(yaw)) + dy * math.cos(math.radians(yaw))
    x = x0 + d_x
    y = y0 + d_y
    z = z0 + dz

    print("Target position: (x: ", x, " y: ", y, " z: ", z, ")")

    # 将位姿保存到CSV列表
    row = [x, y, z, 0, 0, yaw]
    rows.append(row)

    # 构造位姿
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, yaw))

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # get numpy array
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)

    # 将图像保存到文件
    filename = f"sample_{i:04d}.png"
    filepath = os.path.join(save_dir, filename)
    airsim.write_png(os.path.normpath(filepath), img_rgb1)

    # 飞向目标位置
    client.moveToPositionAsync(pose.position.x_val, pose.position.y_val, pose.position.z_val, 5).join()

# release control
client.enableApiControl(False)
# 将位姿列表保存到CSV文件
with open(os.path.join(save_dir, "poses.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
