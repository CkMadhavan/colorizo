from flask import Flask , render_template , request , send_file

app = Flask(__name__)

import keras
from PIL import Image
import numpy as np
import io
from random import randint
import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


@app.route('/')
def index():
    return render_template('up.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST' or request.method == 'GET':
       
      input_dim = 100
    
      f = request.files['file']
      uploaded_files = request.files.getlist("file")
    

      infer = []
      
      
      for f in uploaded_files:
        
        f = Image.open(f).convert('RGB')
        f = f.resize((input_dim,input_dim))
        X_test = np.asarray(f)
        X_test = X_test/225.0

        X_test_gray = rgb2gray(X_test)
        print(X_test_gray.shape)
        X_test_gray = X_test_gray/225.0
        X_test_gray = X_test_gray.reshape(-1,100,100 , 1)

      model = keras.models.load_model('weight1.h5')
      imgs = model.predict(X_test_gray).reshape(100,100,3)

      im = Image.fromarray(np.uint8(imgs * 255.0), 'RGB')
      im = im.resize((500, 500), Image.ANTIALIAS)
      data = io.BytesIO()
      im.save(data, format="png")
      keras.backend.clear_session()
    
      data.seek(0)
      return send_file(data, mimetype='image/png' , cache_timeout=0)

@app.route('/help')
def hel():
    return render_template('help.html')    

if __name__ == "__main__":
    app.run()
