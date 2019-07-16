from flask import Flask , render_template , request

app = Flask(__name__)

import keras
from PIL import Image
import numpy as np
from random import randint

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


@app.route('/')
def index():
    return render_template('up.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       
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

      model = keras.models.load_model('/home/madhav/Downloads/pneumonia-master/weight1.h5')
      imgs = model.predict(X_test_gray).reshape(100,100,3)
      
      i = str(randint(0,100000000))

      im = Image.fromarray(np.uint8(imgs * 255.0), 'RGB')
      string = "/home/madhav/Downloads/pneumonia-master/static/image" + i +".png"
      im.save(string)
      string1 = "image" + i +".png"
      print(string1)
      keras.backend.clear_session()
    
      return render_template('out.html' , string2 = string1)

@app.route('/help')
def hel():
    return render_template('help.html')    

if __name__ == "__main__":
    app.run()
