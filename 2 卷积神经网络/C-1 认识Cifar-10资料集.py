import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

print(X_train[0])  # 显示第一张图片的NumPy数组(32,32,3)
print(Y_train[0])  # 显示对应图片的标签资料，执行结果是6，Cifar10的标签0-9共10个

plt.imshow(X_train[0], cmap="binary")  # 显示这张图片
plt.title("Label: " + str(Y_train[0]))
plt.axis("off")
plt.show()

sub_plot = 330
for i in range(0, 9):  # 显示前9张图
    ax = plt.subplot(sub_plot + i + 1)  # 330表示3行3列索引为0，第一个索引应该为1
    ax.imshow(X_train[i], cmap="binary")
    ax.set_title("Label: " + str(Y_train[i]))
    ax.axis("off")
plt.subplots_adjust(hspace=.5)
plt.show()
