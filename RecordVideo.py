import cv2
import os


cap = cv2.VideoCapture(0)

set_dimension = {
    '480p': (640, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4k': (3840, 2160)
}

filename = 'recorded.avi'
fps = 24.0
my_res = '720p'

Video_Type = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID')
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in Video_Type:
        return Video_Type[ext]

    return Video_Type['avi']



def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


def get_dimensions(cap, res='720p'):
    width, height = set_dimension['480p']
    if res in set_dimension:
        width, height = set_dimension[res]

    change_res(cap, width, height)
    return width, height


dims = get_dimensions(cap, res='480p')
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, fps, dims)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break