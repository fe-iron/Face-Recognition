import cv2
from utility import image_resize, CFEVideoConf
import numpy as np

cap = cv2.VideoCapture(0)
savepath = 'captured/watermark.avi'
fps = 24.0
config = CFEVideoConf(cap,filepath=savepath, res='480p')

out = cv2.VideoWriter(savepath, config.video_type, fps, config.dims)

img_path = 'images/victorylogo.png'
logo = cv2.imread(img_path, -1)
#cv2.imshow('watermark', logo)
watermark = image_resize(logo, height=50)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
#cv2.imshow('watermark', watermark)


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)


    # start_cord_x = 50
    # start_cord_y = 150
    # color = (255, 0, 0)
    # w = 100
    # h = 200
    # stroke = 2
    # end_cord_x = start_cord_x + w
    # end_cord_y = start_cord_y + h
    # cv2.rectangle(frame, (start_cord_x, start_cord_y), (end_cord_x, end_cord_y), color, stroke)

    frame_height, frame_width, frame_c = frame.shape
    # print(frame.shape)
    # cv2.imshow('frame', frame)
    overlay = np.zeros((frame_height, frame_width, 4), dtype='uint8')
    # overlay[100:250, 100:125] = (255, 255, 0, 1) #B G R A
    # overlay[100:250, 150:250] = (0, 255, 0, 1)
    #

    # cv2.imshow('frame', frame)
    watermark_h, watermark_w, watermark_c = watermark.shape

    for i in range(watermark_h):
        for j in range(watermark_w):
            if watermark[i,j][3] != 0:
                h_offset = frame_height - watermark_h
                w_offset = frame_width - watermark_w
                overlay[h_offset + i, w_offset + j]  = watermark[i, j]

    cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()