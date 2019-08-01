from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, render_template, request
from PIL import Image as PILImage

# Define a flask app
app = Flask(__name__)


learn = load_learner('','export.pkl')

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
	
def model_predict(img):
    img = open_image(BytesIO(img))
    pred = learn.predict(img)
    pre = int(pred[0])
    img_data = encode(img)
    result = {"class":pre, "image":img_data}
    return render_template('result.html', result=result)
   

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
        # Make prediction
            preds = model_predict(img)
            return preds
    return 'OK'
	
@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'
    

if __name__ == '__main__':
    app.run()