import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 类似matplotlib

df = pd.read_csv("R-3 iris_data.csv")

# 探索资料

print(df.shape)
print(df.head())
print(df.describe())

# 显示视觉化表图

target_mapping = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}
Y = df["target"].map(target_mapping)
colmap = np.array(["r", "g", "y"])  # 色彩对照表
plt.figure(figsize=(10, 5))  # 图像的长宽
plt.subplot(1, 2, 1)  # 整体图中的位置，即：共一行2列，这个图在索引第一个图
plt.subplots_adjust(hspace=.5)  # wspace、hspace分别表示子图之间左右、上下的间距
plt.scatter(df["sepal_length"], df["sepal_width"], color=colmap[Y])  # 绘制散点图
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.subplot(1, 2, 2)
plt.scatter(df["petal_length"], df["petal_width"], color=colmap[Y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

sns.pairplot(df, hue="target") # 成对显示图片
plt.show()  # pycharm中需要添加这一行
