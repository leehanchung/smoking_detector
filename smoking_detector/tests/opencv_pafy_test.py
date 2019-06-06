# importing video processing libraries
# Must compile and install opencv-python, won't work if pip install opencv-python
import cv2
import pafy
import youtube_dl
import numpy as np

# Load Videos
url = "https://youtu.be/dQw4w9WgXcQ"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture()
cap.open(best.url)

while True:
    # read the VideoCapture, first field is True or False, second field is image in numpy
    ret, image_np = cap.read()

    if image_np is not None:
        cv2.imshow('Object Detection',cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        print("Video Ended")
        break

cap.release()
cv2.destroyAllWindows()
