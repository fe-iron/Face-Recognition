import cv2

cap = cv2.VideoCapture(0)


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)


make_480p()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('frame2', frame)
    cv2.imshow('gray2', gray)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()