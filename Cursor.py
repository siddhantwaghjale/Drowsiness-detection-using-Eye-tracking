import numpy as np
import cv2
import mouse


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 15)
    max_area = 0
    fc = [0, 0, 0, 0]
    for (x,y,w,h) in faces:
        if w*h > max_area:
            max_area = w*h
            fc = [x, y, w ,h]
            
    cv2.rectangle(img,(fc[0],fc[1]),(fc[0]+fc[2],fc[1]+fc[3]),(0,255,0),2)
    roi_gray = gray[fc[1]:fc[1]+fc[3], fc[0]:fc[0]+fc[2]]
    roi_color = img[fc[1]:fc[1]+fc[3], fc[0]:fc[0]+fc[2]]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        xCentre = int((2*ex+ew)/2)
        yCentre = int((2*ey+eh)/2)
        cv2.circle(roi_color,(xCentre,yCentre),10,(0,0,255),1)
        mouse.move(xCentre,yCentre,absolute=True,duration=0)
    cv2.imshow('Detected',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
