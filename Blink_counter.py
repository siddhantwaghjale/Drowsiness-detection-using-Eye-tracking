import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
count1 = [0]
count = count1[0]
status = "OPEN"

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_f = 0

    for (x1,y1,w1,h1) in faces:
        if w1*h1 >= max_f:
            x = x1
            y = y1
            w = w1
            h = h1

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
        
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 14)
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.765, 40)    #Detects smile. Keep smiling!! :)))

    for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if len(eyes) == 0:
        if status == "OPEN":
            count += 1
        status = "CLOSED"
        cv2.putText(img, 'CLOSED',(int(x-w/4), int(y+h/2)), font, 4, (255, 0, 255))
    else:
        if status == "CLOSED":
            count += 1
        status = "OPEN"
        cv2.putText(img, 'OPEN', (int(x-w/4), int(y+h/2)), font, 4, (255, 0, 255))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.putText(img, 'BLINKS:'+ str(int(count/2)),(int(x-w/4), int(y+h/2)+1), font, 2, (0, 0, 0))

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
