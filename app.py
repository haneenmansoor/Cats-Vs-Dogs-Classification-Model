from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.applications.resnet import preprocess_input
#from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img, img_to_array
import os

app = Flask(__name__)

model = load_model('model.h5')

# Set max size of file as 10MB
app.config['MAX_CONTENT_LENGTH'] = 10*1024*1024

# Allow files with extension png, jpg, jpeg
allowed_ext = ['png', 'jpg', 'jpeg']

#def allowed_file(filename):
    #return '.' in filename and\
       #filename.rsplit('.',1)[1].lower() in allowed_ext
    #print(filename)

# function to load image and prepare 
def read_img(filename):
    img = load_img(filename, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
   if request.method == 'POST':
      file = request.files['file']
      try:
         filename=file.filename
         file_path = os.path.join('static/images',filename)
         file.save(file_path)
         img = read_img(file_path)

         preds = model.predict(img)[0]
         th = np.argmax(preds)

         if preds[0] > 0.5:
            prediction = 'cat'
         elif preds[1] > 0.5:
            prediction = 'dog'
         else:
            prediction = 'invalid'
         return render_template('predict.html', animal=prediction, user_image=file_path)
      except Exception as e:
         return 'Unable to read the file. Please check if the file extension is correct.'
   return render_template('predict.html')


if __name__=='__main__':
   app.run(debug=True)