import cv2
import math 
import argparse

from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

import tkinter as tk
from tkinter import *
import tkinter.messagebox as tmsg



def highlightFace(net,frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def check():
     video=cv2.VideoCapture(args.image if args.image else 0)
     padding=20
     frame=Toplevel(window)
     while cv2.waitKey(1)<0:

         hasFrame,frame=video.read()
         if not hasFrame:
             cv2.waitKey()
             break

         resultImg,faceBoxes=highlightFace(faceNet,frame)
         if not faceBoxes:
             print("No face detected")
             a=tmsg.showinfo("No Face Detected")

         for faceBox in faceBoxes:
             face=frame[max(0,faceBox[1]-padding):
                     min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                     :min(faceBox[2]+padding, frame.shape[1]-1)]

             blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
             genderNet.setInput(blob)
             genderPreds=genderNet.forward()
             gender=genderList[genderPreds[0].argmax()]
             print(f'Gender: {gender}')

             ageNet.setInput(blob)
             agePreds=ageNet.forward()
             age=ageList[agePreds[0].argmax()]
             print(f'Age: {age[1:-1]} years') 

         labels = []
         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
         faces = face_classifier.detectMultiScale(gray,1.3,5)

         for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(6,204,197),2)
             roi_gray = gray[y:y+h,x:x+w]
             roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


             if np.sum([roi_gray])!=0:
                 roi = roi_gray.astype('float')/255.0
                 roi = img_to_array(roi)
                 roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class

                 preds = classifier.predict(roi)[0]
                 print("\nprediction = ",preds)
                 label=class_labels[preds.argmax()]
                 print("\nprediction max = ",preds.argmax())
                 print("\nlabel = ",label)
                 label_position = (x,y)
                 cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(6,204,197),3)
             else:
                 cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(6,204,197),3)
             print("\n\n")
            



             cv2.putText(resultImg, f'{gender}, {age}, {label}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (290,150,230), 2, cv2.LINE_AA)
             cv2.imshow("Detecting age and gender", resultImg)
            

  






parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
genderList=['Male','Female']
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']



faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


window = tk.Tk()
window.title("G5 Group")
#window.config(background='white')
window.geometry("794x480")

l1 = tk.Label(window,text="PRIYADARSHINI COLLEGE OF ENGINEERING",font="Algerian 30",fg='blue',pady=20)
l1.grid(column=0, row=0)

l2 = tk.Label(window,text="Age Gender & Emotion Detection",font="Algerian 27",fg='black',pady=20)
l2.grid(column=0, row=1)

b1 = tk.Button(window, text="DETECT THE FACE", font="Algerian 20",bg='green', fg='white', padx=20,command=check).place(x=260,y=200,width=250,height=50)
#b1.grid(column=0, row=4) 



window.mainloop()








