# Apply ResNet model to predict objectory
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# Prepare image
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_arry(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction for the image
preds = model.predict(x)

print("Predicted: ", decode_predictions(preds, top=3)[0])
