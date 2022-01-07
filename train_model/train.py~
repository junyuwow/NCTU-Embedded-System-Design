import cv2 as cv2
import numpy as np
from os import listdir

images = []
labels = []
label_1_path = "1/"
label_2_path = "2/"
label_3_path = "test/"
files = listdir(label_1_path)
for f in files:
  images.append(cv2.resize(cv2.imread(label_1_path+"/"+f,0),(55,55)))
  labels.append(1)

files = listdir(label_2_path)
for f in files:
  images.append(cv2.resize(cv2.imread(label_2_path+"/"+f,0),(55,55)))
  labels.append(2)

recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=150)
recognizer.train(images,np.array(labels))
recognizer.save("10_picture_model.xml")
'''
predict
target = cv2.resize(cv2.imread("t01_i.png",0),(55,55))
label,confidence=recognizer.predict(target)
print(label,confidence)
'''
