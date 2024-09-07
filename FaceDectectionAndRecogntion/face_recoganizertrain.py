import os
import cv2 as cv
import numpy as np

people = ['Dulquer']
DIR = r"C:\Users\hp\Downloads\faceRecoganizertrain"
features = []
labels = []
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Define a fixed size for all face images
face_size = (100, 100)

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Error reading image {img_path}. Skipping.")
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                # Resize face image to a fixed size
                faces_roi = cv.resize(faces_roi, face_size)
                features.append(faces_roi)
                labels.append(label)

create_train()
print(f'Length of features: {len(features)}')
print(f'Length of labels: {len(labels)}')

# Convert lists to NumPy arrays
features = np.array(features, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer with features list and labels list
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
