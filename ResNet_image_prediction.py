# -*- coding: utf-8 -*-
import os
import cv2
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# USAGE
# python ResNet_image_prediction.py --image ./Dataset/test/IMG_13.jpg

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image we are going to classify")
args = vars(ap.parse_args())

model = load_model('./the_save_model.h5')
print("loading network...")

classes = [ 'owl', # 0
            'galaxy', # 1
            'lightning', # 2
            'wine-bottle', # 3
            't-shirt', # 4
            'waterfall',# 5
            'sword', # 6
            'school-bus',# 7           
            'calculator', # 8
            'sheet-music', #9            
            'airplanes',#10      
            'lightbulb', # 11
            'skyscraper',#12    
            'mountain-bike',#13          
            'fireworks', #14        
            'computer-monitor',#15              
            'bear',# 16
            'grand-piano', # 17
            'kangaroo', # 18
            'laptop', #19
           ]

#model.summary()

# load the input image and resize it to the target spatial dimensions
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (256, 256))

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = classes[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
