import cv2
import numpy as np
from PIL import Image


recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainingData.yml")
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


path="dataset"
cam=cv2.VideoCapture(0)
id=0
font =cv2.FONT_HERSHEY_SCRIPT_COMPLEX
while True:
    ret,face =cam.read()
    gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.6,1)
    for (x,y,w,h) in faces:
        cv2.rectangle(face,(x,y),(x+w, y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face[y:y+h, x:x+w]
        
        
        id ,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if (id == 1):
            id = 'Jisan'
        elif(id == 2):
            id = 'Zinnah'
        elif(id == 3):
            id = 'Reshma'
        elif(id == 4):
            id = 'Nafis Sir'
        elif(id == 5):
            id = 'Donald Trump'
        elif(id == 6)
            id = 'Ikhtiar'
        else:
            id ='Unknown'
        cv2.putText(face, str(id), (x,y+h+50), font, 2, (255,255,255))
    cv2.imshow('frame',face)   
    if (cv2.waitKey(1)&0xff == ord('q')):
          break
cam.release()
cv2.destroyAllWindows()
