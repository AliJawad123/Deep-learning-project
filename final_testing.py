import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from tensorflow.keras.models import load_model
from model_function import resize,detect
import imageio as iio
import os
from pygame import mixer
import time


def make_pred(mymodel):
    detect_and_resize()
    drowsy=0
    for i in os.listdir('//home//jawad//to_jawad//resized_to_predict'):
        img=iio.imread('//home//jawad//to_jawad//resized_to_predict//'+i)
        img2=np.expand_dims(img,axis=0)
        pred=mymodel.predict(img2)
        classified=classify(pred)
        if(classified==1):
            drowsy+=1
    
    print('drowsy = ',drowsy)
    length=len(os.listdir('//home//jawad//to_jawad//resized_to_predict'))
    print('ratio = ',drowsy/length)
    start=time.time()
    if(drowsy/length>0.2):
        end=time.time()
        mixer.init()
        mixer.music.load('song.mp3')
        mixer.music.play()
        while(end-start<10):
          #  print('running')
            end=time.time()
        mixer.music.stop()
            
            
            

        
        
        

def classify(pred):
    return np.argmax(pred)


def detect_and_resize():
    detect()
    resize()


mymodel=load_model('final_model.h5')
#make_pred(mymodel)


'''


drowsy=0
for i in os.listdir('//home//jawad//to_jawad//resized_to_predict'):
        img=iio.imread('//home//jawad//to_jawad//resized_to_predict//'+i)
        img2=np.expand_dims(img,axis=0)
        print('for ',i,' = ')
        pred=mymodel.predict(img2)
        classified=classify(pred)
        if(classified==1):
            drowsy+=1
        print(classified)
print(drowsy)
    

'''




#print(mymodel.summary())


'''
detect()
resize()
img=iio.imread('resized_to_predict//detected_face_2.jpg')
img2=np.expand_dims(img,axis=0)
pred=mymodel.predict(img2)
print('pred = ',pred)
print('argmax = ',)
'''