from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions

# 下载不行需要把H5文件放在Windows系统的缓存文件夹下：C:\Users\用户名.keras\models

model = InceptionV3(weights="imagenet", include_top=True)

img = load_img("R-1 koala.png", target_size=(299, 299))  # 不同之处是这里的size是(299,299)
x = img_to_array(img)
print("x.shape: ", x.shape)
img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

img = preprocess_input(img)
print("img.shape: ", img.shape)

# 预测

Y_pred = model.predict(img)
label = decode_predictions(Y_pred)
result = label[0][0]
print("%s (%.2f%%)" % (result[1], result[2] * 100))
