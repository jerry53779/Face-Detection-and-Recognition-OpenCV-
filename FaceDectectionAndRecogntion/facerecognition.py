import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Dulquer']

features = np.load('features.npy')
labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img_path = r"C:\Users\hp\Downloads\faceRecoganizertrain\Dulquer\download (1).jpeg"
img = cv.imread(img_path)

if img is None:
    print(f"Error loading image {img_path}.")
else:
    cv.imshow('Original', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_dect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_dect:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv.resize(face_roi, (100, 100))  
        label, confidence = face_recognizer.predict(face_roi)
        print(f'Label={label}, Confidence={confidence}')
        cv.putText(img, str(people[label]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Detected Face', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
