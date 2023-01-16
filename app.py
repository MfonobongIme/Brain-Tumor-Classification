import os
import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded')

#define 
def get_className(ClassNo):
    if ClassNo == 0:
        return 'No Brain Tumor'
    elif ClassNo == 1:
        return 'Yes Brain Tumor'

#get result function
def getResult(img):
    image = cv2.imread(img) #read image
    image = Image.fromarray(image, 'RGB') #convert to grayscale
    image = image.resize((64, 64)) #resize
    image = np.array(image) #convert to numpy array
    input_img = np.expand_dims(image, axis = 0) #add one more dimension for our model
    result = model.predict(input_img) #predict
    return result

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    #f its a post method
    if request.method == 'POST':
       f = request.files['file']

       basepath = os.path.dirname(__file__)
       file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
       f.save(file_path)
       value = getResult(file_path)
       result = get_className(value)
       return result
    return None #incase of errors

if __name__ == '__main__':
    app.run(debug = True)





