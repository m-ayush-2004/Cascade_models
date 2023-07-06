import cv2
import numpy as np
cap=cv2.VideoCapture(0); 

harr_eye=cv2.CascadeClassifier(r'cascade_vision\haarcascade_eye.xml')

harr_face=cv2.CascadeClassifier(r'cascade_vision\haarcascade_frontalface_alt.xml')

harr_fb=cv2.CascadeClassifier(r'cascade_vision\haarcascade_fullbody.xml')

harr_lb=cv2.CascadeClassifier(r'cascade_vision\haarcascade_lowerbody.xml')

harr_smile=cv2.CascadeClassifier(r'cascade_vision\haarcascade_smile1.xml')

harr_ub=cv2.CascadeClassifier(r'cascade_vision\haarcascade_upperbody.xml')
while True:
    ret,frame=cap.read()
    frame1= cv2.resize(frame,(300,300))
    gray1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray3=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray4=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray5=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray6=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    eyes=harr_eye.detectMultiScale(gray1,1.2,4)
    face=harr_face.detectMultiScale(gray2,1.2,4)
    smile=harr_smile.detectMultiScale(gray3,1.1,minNeighbors=18,minSize=(25, 25))
    ub=harr_ub.detectMultiScale(gray4,1.2,4)
    lb=harr_lb.detectMultiScale(gray5,1.2,4)
    fb=harr_fb.detectMultiScale(gray6,1.2,4)
    
    for (x,y,w,h) in eyes:
        cv2.rectangle(gray1,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in face:
        cv2.rectangle(gray2,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in smile:
        cv2.rectangle(gray3,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in ub:
        cv2.rectangle(gray4,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in lb:
        cv2.rectangle(gray5,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in fb:
        cv2.rectangle(gray6,(x,y),(x+w,y+h),(0,255,255),3)

    cv2.imshow("orignal_image",frame1)
    cv2.imshow("eyes_image",gray1)
    cv2.imshow("face_image",gray2)
    cv2.imshow("smile_image",gray3)
    cv2.imshow("ub_image",gray4)
    cv2.imshow("lb_image",gray5)
    cv2.imshow("fb_image",gray6)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
