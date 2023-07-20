import airsim
import os
import numpy as np

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取图像路径
folder_path = "E:/FunctionMethod/airsim_images"  # 保存图像的文件夹路径
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 设置随机采样的范围和数量
num_samples = 1000  # 需要采样的数量
x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -20, -10  # 采样范围

# 随机采样并保存图像和位姿信息
poses = []
for i in range(num_samples):
    # 随机生成目标位置，并设置姿态朝向正向
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    poses.append(pose)

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # get numpy array
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)

    # 保存图像和位姿信息
    img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.png".format(i, x, y, z)
    img_filepath = os.path.join(folder_path, img_filename)
    airsim.write_png(os.path.normpath(img_filepath), img_rgb1)

    pose_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.txt".format(i, x, y, z)
    pose_filepath = os.path.join(folder_path, pose_filename)
    with open(pose_filepath, 'w') as f:
        f.write("{0}\n".format(pose))

print("全部图像和位姿信息均已保存到文件夹：", folder_path)
