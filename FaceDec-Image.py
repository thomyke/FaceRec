import numpy as np
import cv2 as cv

# Read image from your local file system
image = cv.imread('C:/your-image1.jpg')

# Convert color image to grayscale for Viola-Jones
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

face_cascade1 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
face_cascade2 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
face_cascade3 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_profileface.xml')

detected_faces1 = face_cascade1.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=6,
                                         minSize=(60, 60),
                                         flags=cv.CASCADE_SCALE_IMAGE)
detected_faces2 = face_cascade2.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=6,
                                         minSize=(60, 60),
                                         flags=cv.CASCADE_SCALE_IMAGE)
detected_faces3 = face_cascade3.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=6,
                                         minSize=(60, 60),
                                         flags=cv.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in detected_faces1:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceROI = image[y:y + h, x:x + w]
    cv.imwrite(str(w) + str(h) + '_faces.jpg', faceROI)

for (x, y, w, h) in detected_faces2:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceROI = image[y:y + h, x:x + w]
    cv.imwrite(str(w) + str(h) + '_faces.jpg', faceROI)

for (x, y, w, h) in detected_faces3:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceROI = image[y:y + h, x:x + w]
    cv.imwrite(str(w) + str(h) + '_faces.jpg', faceROI)


cv.imshow('Image', image)

cv.imshow('ImageGray', gray)

cv.waitKey(0)
cv.destroyAllWindows()
