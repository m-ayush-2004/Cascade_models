import cv2
import numpy as np
cap=cv2.VideoCapture(0);
csc=cv2.CascadeClassifier('tutorial cv\\cars.xml')
csm=cv2.CascadeClassifier('tutorial cv\\haarcascade_eye_tree_eyeglasses.xml')
while True:
    ret,frame=cap.read()
    frame1= cv2.resize(frame,(600,600))
    gray=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces=csm.detectMultiScale(gray,1.2,4)
    eyes=csc.detectMultiScale(gray,1.2,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,255),3)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
    frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame",frame1)
    cv2.imshow("frame2",frame)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
