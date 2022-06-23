import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img

# 显示图片和类型

img = load_img("R-3 photo.png")

print(type(img))
print(img.format)
print(img.mode)
print(img.size)

plt.axis("off")
plt.imshow(img)
# plt.show()

# 将图片转换成NumPy阵列

img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)

# 将NumPy阵列转换为图片

img2 = array_to_img(img_array)
print(type(img2))

# 改为灰阶图片、调整图片尺寸

img3 = load_img("R-3 photo.png", color_mode="grayscale", target_size=(227, 227))
plt.imshow(img3)
# plt.show()

# 保存图片

img = load_img("R-3 photo.png", color_mode="grayscale")
img_array = img_to_array(img)
# save_img("photo_gray.jpg", img_array)

# 图片增强

img = load_img("R-3 photo.png")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)  # 重塑成(1,320,320,3)的格式
print(x.shape)

datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转，最多旋转40度
    width_shift_range=0.2,  # 随机位移，以图片中心点随机位移，长宽最多可以位移20%
    height_shift_range=0.2,
    shear_range=0.2,  # 随机推移变换（图片视觉上不在水平线上，如长方形变平行四边形），强度0.2
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True  # 水平方向随机翻转，垂直：vertical
)

i = 0
for batch_img in datagen.flow(x, batch_size=1,  # x是原始图片，二参是每批次产生几张图片
                              save_to_dir="R-2 preview",  # 保存到的目录，需要事先建立
                              save_prefix="pen",  # 生成图片的字头
                              save_format="jpeg"):
    plt.axis("off")
    plt.imshow(batch_img[0].astype("int"))
    plt.show()
    i += 1
    if i >= 10:  # 控制生成10张图片
        break
