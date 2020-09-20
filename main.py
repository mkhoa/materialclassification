from flask import Flask, render_template, render_template, request, jsonify, make_response

import tensorflow as tf
import numpy as np
import re
import os
import base64
import uuid


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home_page.html') 


model = tf.keras.models.load_model('Inception_15_class.h5')

label_map = np.array(['brick', 'ceramic', 'fabric', 'glass', 'leather', 'metal',
       'mirror', 'painted', 'paper', 'plastic', 'polishedstone', 'stone',
       'tile', 'wallpaper', 'wood'])

IMG_WIDTH = 192
IMG_HEIGHT = 192

def read_raw_img(image):
    image = tf.image.central_crop(image, 0.55)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image
    
@app.route('/predict/', methods=['POST'])
def predict():
    ''' Arg: array of image of size 192, 192, 3
        Returns: list of predict labels
    '''
    data = request.get_json()
    img_raw = data['data_uri'].encode()
    image = read_raw_img(img_raw)
    
    result = model.predict(image)
    pred_tag=[]
    
    for i in range(15):
        prob = result[0][i]
        label = label_map[i]
        if(prob == max(result[0])):
            pred_tag.append(label)
        prediction = str(' '.join(pred_tag))
            
    return jsonify({'label': prediction, 'probs': max(result[0])}) 


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
  
