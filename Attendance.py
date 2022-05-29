import cv2
import numpy as np
import face_recognition
import os
import PluginsPy
from datetime import datetime

path = 'venv/Imagebasic'
images = []  #initialising list
Names = []
list = os.listdir(path)
print(list)
#creating loop so that we don't need to read every file individually
for i in list:
    img = cv2.imread(f'{path}/{i}')
    images.append(img)
    Names.append(os.path.splitext(i)[0])
print(Names)
#for encoding, write a funcn
def encodings(images):
    encodinglist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodinglist.append(encode)
    return encodinglist

#calling funcn and printing length of that to know number of images encoded
encodes = encodings(images)
print(len(encodes))
#To take input through webcam
vid = cv2.VideoCapture(0)
while True:

    success, img = vid.read()
    #reducing size of image
    img1 = cv2.resize(img,(0,0),None,0.25,0.25)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img1)
    encodingfaces = face_recognition.face_encodings(img1,faces)

#To find matches

    for encodingface,faceloc in zip(encodingfaces,faces):
        match = face_recognition.compare_faces(encodes,encodingface)
        dist = face_recognition.face_distance(encodes,encodingface)

        print(dist)
        Index = np.argmin(dist)

        if match[Index]:
            name = Names[Index].upper()
            print(name)


        # defining a function to mark attendance
        def markattendance(name):
            with open('AttendanceInfo.csv', 'r+') as f:
                attlist = f.readlines()
                nameslist = []  # initialising an empty list
                for line in attlist:
                    person = line.split(',')
                    nameslist.append(person[0])  # appending nameslist
                # If a name is not present in list,we write their names and time of joining,if present..no changes
                if name not in nameslist:
                    now = datetime.now()
                    dstring = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dstring}')

        y1,x2,y2,x1 = faceloc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)
        markattendance(name)
        cv2.imshow('Cam',img)
        cv2.waitKey(0)