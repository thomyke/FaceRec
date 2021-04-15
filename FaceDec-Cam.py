import cv2 as cv
import time 
from datetime import datetime

# loadnutie cascad z xml opencv
faceCasMy = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_MyCustom.xml')


# recognizer = cv.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()

# detector = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

# recognizer.train(faces, np.array(Id))
# recognizer.save("TrainingImageLabel\Trainner.yml")
# res = "Image Trained"#+",".join(str(f) for f in Id)
# message.configure(text= res)

# faceCascade2 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\MyCustomKaskade.xml')
# faceCascade2 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\TrainKaskadeTomas.xml')
# faceCascade2 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\Front_Prof_KaskadeCustomTomas.xml')
faceCascade1 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
faceCascade2 = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
faceCascadeD = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
faceCasProfil = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_profileface.xml')
eyeCascade = cv.CascadeClassifier('C:\DEV\work\projekt\Lib\site-packages\cv2\data\haarcascade_eye.xml')
video_capture = cv.VideoCapture(0)

star_time = time.time()
#pokial ide video snímaj
while True:
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    ret, frame = video_capture.read()
    gray = cv.GaussianBlur(frame,(7,7),0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces1 = faceCascade1.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=8,
                                         minSize=(60, 60),
                                         flags=cv.CASCADE_SCALE_IMAGE)
    faces2 = faceCascade2.detectMultiScale(gray,
                                           scaleFactor=1.2,
                                           minNeighbors=5,
                                           minSize=(50, 50),
                                           flags=cv.CASCADE_SCALE_IMAGE)
    facesProfil = faceCasProfil.detectMultiScale(gray,
                                           scaleFactor=1.3,
                                           minNeighbors=6,
                                           minSize=(60, 60),
                                           flags=cv.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces1:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        faceROI = frame[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        end_time = time.time()
        print(end_time)
        duration = end_time - star_time
        print("----------------------------------")
        print(f'Cas od spustenia :  {round(duration, 3)}')
        print(f'Cas zaznamu:  {dtString}')
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

            end_time = time.time()
            print(end_time)
            duration = end_time - star_time
            print("----------------------------------")
            print(f'Cas od spustenia :  {round(duration, 3)}')
            print(f'Cas zaznamu:  {dtString}')
        cv.imwrite(str(dtString) + str(h) + '_faces2.jpg', faceROI)
    for (x, y, w, h) in faces2:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        faceROI = frame[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        end_time = time.time()
        print(end_time)
        duration = end_time - star_time
        print("----------------------------------")
        print(f'Cas od spustenia :  {round(duration, 3)}')
        print(f'Cas zaznamu:  {dtString}')
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

            end_time = time.time()
            print(end_time)
            duration = end_time - star_time
            print("----------------------------------")
            print(f'Cas od spustenia :  {round(duration, 3)}')
            print(f'Cas zaznamu:  {dtString}')
        cv.imwrite(str(w) + str(h) + '_faces3.jpg', faceROI)
    for (x, y, w, h) in facesProfil:
	
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        faceROI = frame[y:y + h, x:x + w]
        end_time = time.time()
        print(end_time)
        duration = end_time - star_time
        print("----------------------------------")
        print(f'Cas od spustenia :  {round(duration, 3)}')
        print(f'Cas zaznamu:  {dtString}')
        cv.imwrite(now.strftime('%H:%M:%S') + '_' + str(round(duration, 2)) + '_faces4.jpg', faceROI)
		
    # zobrazenie videa + vypnutie
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv.destroyAllWindows()
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
	#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)
