import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
images_path = 'D:\Files\python\Face Recognition\venv\images\Jun_Ji_Hyun'
#Read the input image

#img = cv2.imread('images/Jun_ji_Hyun/2.jpg')
cap = cv2.videoCapture("filename")

while cap.isOpened():
    img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    #Display the output
    cv2.imshow('img', img)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break


cap.release()