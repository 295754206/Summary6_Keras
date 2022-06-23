import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


# 打乱资料顺序

def randomize(a, b):
    permutation = list(np.random.permutation(a.shape[0]))  # 产生随机索引清单值
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]

    return shuffled_a, shuffled_b


X_train, Y_train = randomize(X_train, Y_train)

# 取出前500张图片

X_train = X_train[:500]  # 这里是大数组
Y_train = Y_train[:500]

# 将训练资料的图片尺寸放大

print("将训练资料的图片尺寸放大... ")

# np.array()和np.asarray()区别：
# 当数据源是ndarray类型如 np.ones((3,3)) 时，array是生成一个与原数据无关的新内存，而asarray是引用

X_train_new = np.array(  # 3.最后转化为大数组
    [
        np.asarray(  # 2.再转化为数组
            Image.fromarray(  # 1.转换为图片
                X_train[i]
            ).resize(
                (200, 200)  # 放大成(200,200)尺寸
            ))
        for i in range(0, len(X_train))
    ]
)

# 绘制前六张图片

fig = plt.figure(figsize=(10, 7))
sub_plot = 230
for i in range(0, 6):
    ax = plt.subplot(sub_plot + i + 1)
    ax.imshow(X_train_new[i], cmap="binary")
    ax.set_title("Label: " + str(Y_train[i]))
plt.show()
