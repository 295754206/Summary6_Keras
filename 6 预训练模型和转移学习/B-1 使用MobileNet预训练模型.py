from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

# 载入预训练模型

model = MobileNet(weights="imagenet", include_top=True)  # ImageNet有100万张图片训练过，分1000种

# 载入测试图片

img = load_img("R-1 koala.png", target_size=(224, 224))  # 调整成(224,224)尺寸
x = img_to_array(img)  # 转换成numpy阵列
print("x.shape: ", x.shape)
img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))  # 格式改为(1, 224, 224, 3)

# 资料预处理

img = preprocess_input(img)  # 将numpy阵列转换成模型所需的输入资料
print("img.shape: ", img.shape)

# 使用模型进行预测

Y_pred = model.predict(img)  # 获取预测输出
label = decode_predictions(Y_pred)  # 查询对应名和对应概率
result = label[0][0]  # 第一个是最可能的结果
print("%s (%.2f%%)" % (result[1], result[2] * 100))  # 显示名称和对应概率
