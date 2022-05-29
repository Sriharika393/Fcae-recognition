import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('venv/Imagebasic/rm_test.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('venv/Imagebasic/rm_test.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
imgtest2 = face_recognition.load_image_file('venv/Imagebasic/chiru.jpg')
imgtest2 = cv2.cvtColor(imgtest2,cv2.COLOR_BGR2RGB)

loc = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]
cv2.rectangle(img,(loc[3],loc[0]),(loc[1],loc[2]),(255,0,255),2)

loc1 = face_recognition.face_locations(imgtest)[0]
encode1 = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(loc1[3],loc1[0]),(loc1[1],loc1[2]),(255,0,255),2)

loc2 = face_recognition.face_locations(imgtest2)[0]
encode2 = face_recognition.face_encodings(imgtest2)[0]
cv2.rectangle(imgtest2,(loc2[3],loc2[0]),(loc2[1],loc2[2]),(255,0,255),2)

result = face_recognition.compare_faces([encode],encode1)
dis = face_recognition.face_distance([encode],encode1)
print(result,dis)

result1 = face_recognition.compare_faces([encode],encode2)
dis1 = face_recognition.face_distance([encode],encode2)
print(result1,dis1)

cv2.putText(imgtest,f'{result} {round(dis[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
cv2.putText(imgtest2,f'{result1} {round(dis1[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

cv2.imshow('RM',img)
cv2.imshow('RM-test',imgtest)
cv2.imshow('Chiru',imgtest2)

cv2.waitKey(0)

