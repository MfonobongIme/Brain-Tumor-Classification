import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5') #load model

#load image path
image = cv2.imread('C:/Users/hp/Desktop/projects_git/brain_tumor_classification/pred/pred0.jpg')
img = Image.fromarray(image)

#resize image
img = img.resize((64, 64)) #resize image
img = np.array(img) #convert to np array

#expand dimension of image
input_img = np.expand_dims(img, axis = 0)

#print(img)

result = model.predict(input_img)
print(result)