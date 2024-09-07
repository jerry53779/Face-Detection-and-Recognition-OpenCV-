import cv2 as cv

img=cv.imread("Images/JERRY-15c47cd9-b0e0-4945-b10e-1d315027d188 (1).jpg")
cv.imshow('Image',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Grayed',gray)

#haar_cascade=cv.CascadeClassifier('haar_face.xml')
#face_dect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
#for (x,y,w,h) in face_dect:
#    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#cv.imshow("Dectected Image",img)
#cv.waitKey(0)

capture=cv.VideoCapture(0)
while True:
    isTrue, frames=capture.read()
    cv.imshow('Video live',frames)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    gray=cv.cvtColor(frames,cv.COLOR_BGR2GRAY)
    haar_cascade=cv.CascadeClassifier('haar_face.xml')
    face_dect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in face_dect:
        cv.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    cv.imshow("Dectected Video",frames)
capture.release()
cv.destroyAllWindows()