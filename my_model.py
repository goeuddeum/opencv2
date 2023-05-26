#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

file_path= os.path.abspath(__file__) #~/catkin_ws/my_cam/scripts/my_model.py
dir_path= os.path.dirname(file_path) #~/catkin_ws/my_cam/scripts/
model_path= os.path.join(dir_path,"..","model","keras_model.h5") #~/catkin_ws/my_cam/model/keras_mdoel.h5
label_path= os.path.join(dir_path,"..","model","labels.txt") #~/catkin_ws/my_cam/model/labels.txt

# Load the model
model = load_model(model_path, compile=False)

# Load the labelsq
class_names = open(label_path, "r").readlines()
#["0 1000", "1 마우스", "2 핸드폰"]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image_og = camera.read()

    if not ret:
        continue

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image_og, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1 # image / 255 *2 -1
    # 0 < x < 255 >>>   -1 < x < 1
    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    #["0 1000", "1 마우스", "2 핸드폰"]
    confidence_score = prediction[0][index]

    if confidence_score < 0.8:
        text = "I don't know"

    # Print prediction and confidence score
    name = class_name[2:]
    result =str(np.round(confidence_score * 100))[:-2] + "%"
    text = name + " " + result
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", result)
    cv2.putText(image_og, text,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 3)
       
    # Show the image in a window
    cv2.imshow("Webcam Image", image_og)



    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break


camera.release()
cv2.destroyAllWindows()

# my_model.py 를 수정하여 예측 결과를 이미지에 출력하기