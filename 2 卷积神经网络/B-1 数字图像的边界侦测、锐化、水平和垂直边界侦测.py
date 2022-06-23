import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

img = np.load("R-1 digit_numb/digit8.npy")
edge = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]
sharpen = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

# 原始图像

plt.figure()  # 表示这是一张图
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")  # 像素显示色
plt.axis("off")  # 不显示标刻尺
plt.title("original image")

# 边界侦测

plt.subplot(1, 3, 2)
c_digit = signal.convolve2d(img, edge, boundary="symm", mode="same")  # 使用edge边界侦测过滤器
plt.imshow(c_digit, cmap="gray")
plt.axis("off")
plt.title("edge-detection image")

# 锐化

plt.subplot(1, 3, 3)
c_digit = signal.convolve2d(img, sharpen, boundary="symm", mode="same")  # 使用edge边界侦测过滤器
plt.imshow(c_digit, cmap="gray")
plt.axis("off")
plt.title("sharpen image")
plt.show()

# 水平和垂直边界侦测

img = np.load("R-1 digit_numb/digit3.npy")
filters = [
    [
        [-1, -1, -1],
        [1, 1, 1],
        [0, 0, 0]
    ],
    [
        [-1, 1, 0],
        [-1, 1, 0],
        [-1, 1, 0]
    ],
    [
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1]
    ],
    [
        [0, 1, -1],
        [0, 1, -1],
        [0, 1, -1]
    ]
]

plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("original")

for i in range(2, 6):
    plt.subplot(1, 5, i)
    c = signal.convolve2d(img, filters[i - 2], boundary="symm", mode="same")
    plt.imshow(c, cmap="gray")
    plt.axis("off")
    plt.title("filter-" + str(i - 1))  # 分别是下方水平、右边垂直、上方水平、左边垂直的边线

plt.show()
