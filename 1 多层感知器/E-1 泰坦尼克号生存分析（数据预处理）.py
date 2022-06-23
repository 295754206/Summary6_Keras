import numpy as np
import pandas as pd

df = pd.read_csv("R-4 titanic_data.csv")

# 探索数据

print(df.shape)
print(df.head())
print(df.describe())  # 发现有的栏目不是1309而是1308，说明有资料缺失

# 发现缺失值

print(df.info())  # 显示每一列资料的详情
print(df.isnull().sum())  # 显示有缺失的资料

# 删除不是特征值的栏位

df = df.drop(["name", "ticket", "cabin"], axis=1)

# 缺失资料填充平均值

df[["age"]] = df[["age"]].fillna(value=df[["age"]].mean())
df[["fare"]] = df[["fare"]].fillna(value=df[["fare"]].mean())

# 名称特征值填充众数

df[["embarked"]] = df[["embarked"]].fillna(value=df["embarked"].value_counts().idxmax())
print(df["embarked"].value_counts())
print(df["embarked"].value_counts().idxmax())

# 转换分类数据

df["sex"] = df["sex"].map({"female": 1, "male": 0}).astype(int)

# embarked栏位的One-hot编码

embarked_one_hot = pd.get_dummies(df["embarked"], prefix="embarked")  # 使用函数拆分该栏位并设置新栏位前缀字符串
df = df.drop("embarked", axis=1)  # 删除老栏位
df = df.join(embarked_one_hot)  # 加入one-hot编码

# 将标签栏移动到最后

df_survived = df.pop("survived")
df["survived"] = df_survived
print(df.head())

# 使用乱数mask分割训练和测试数据集

mask = np.random.rand(len(df)) < 0.8  # 使用乱数产生80%的mask变数（80%是true，20%是false的数组）
df_train = df[mask]
df_test = df[~mask]
print("Train: ", df_train.shape)
print("Test:  ", df_test.shape)

# 存储处理过后的数据

df_train.to_csv("R-4-1 titanic_train.csv", index=False)
df_test.to_csv("R-4-2 titanic_test.csv", index=False)
