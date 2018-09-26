# -*- coding: utf-8 -*-
"""
Created on Wed May  30 12:29:43 2018

@author: Ayush
"""








from skimage import io
from sklearn.cluster import KMeans
import numpy as np

from flask import Flask, render_template, request
import numpy as np
import cv2
import imageio
import requests
import cv2
import json
import os
from sklearn.cluster import MiniBatchKMeans
app =Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))




def detect(image):
	
    (r, c) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)#better to convert to a format that works better with euclidean distances
    image = image.reshape((r * c, 3))#reshaping the image
    clusters = MiniBatchKMeans(n_clusters = 8)#applying kmeans to extract dominant colors
    category = clusters.fit_predict(image)
    reconstructed_image = clusters.cluster_centers_.astype("uint8")[category]#get centers for each pixel and recreate the whole image
    reconstructed_image = reconstructed_image.reshape((r, c, 3))
    image = image.reshape((r, c, 3))
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return(reconstructed_image)

    	
    


    



##rest of these functions are simply to read an image and return a result on webpage and would be pretty much obvious 
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():    
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    
    image= cv2.imread(destination)
									
    frame=detect(image)#doinfg predictions on the image
    
    
    cv2.imwrite('static/'+filename,frame)##writing the image and loading it on html gallery
    
    return render_template("results.html",image_name=filename)
    
    
if __name__ == "__main__":
    print(("Loading"))
    
    app.run()
