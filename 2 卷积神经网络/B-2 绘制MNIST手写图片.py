import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])  # 这是数字5的阵列图
print(Y_train[0])  # 标签声明是5

# 绘制图像5

plt.imshow(X_train[0], cmap="gray")
plt.title("Label:" + str(Y_train[0]))
plt.axis("off")
plt.show()

# 绘制前9张图片

sub_plot = 330

for i in range(0, 9):
    ax = plt.subplot(sub_plot + i + 1)  # 331表示图像是3X3格式中的第1张图片
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title("Label:" + str(Y_train[i]))
    ax.axis("off")

plt.subplots_adjust(hspace=.5)
plt.show()
