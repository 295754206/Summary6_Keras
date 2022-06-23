from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions

model = ResNet50(weights="imagenet", include_top=True)

img = load_img("R-1 koala.png", target_size=(224, 224))
x = img_to_array(img)
print("x.shape: ", x.shape)
img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

img = preprocess_input(img)
print("img.shape: ", img.shape)

# 预测

Y_pred = model.predict(img)
label = decode_predictions(Y_pred)
result = label[0][0]  
print("%s (%.2f%%)" % (result[1], result[2]*100))