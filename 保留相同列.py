import pandas as pd

# 读取csv文件
df1 = pd.read_csv("E:/FunctionMethod/airsim_images/desert_datasets/circular_10k9/image_dataset.csv")
df2 = pd.read_csv("E:/FunctionMethod/airsim_images/desert_datasets/poses.csv")

# 合并两个dataframe
merged_df = pd.merge(df1, df2, on="index")

# 保留合并后的dataframe中的数据
result_df = merged_df.drop("index", axis=1)

# 将结果保存到csv文件中
result_df.to_csv("E:/FunctionMethod/airsim_images/desert_datasets/circular_10k9/result.csv", index=False)

