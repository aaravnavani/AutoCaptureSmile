from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2

#get MOUTH ASPECT RATIO (MAR) value 
def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar


counter = 0
total = 0


shape_predictor= "shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

fps= FPS().start()
cv2.namedWindow("Video Stream")

counter1 = -1
def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    print("testing")

while True:
    frame = vs.read()    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth= shape[mStart:mEnd]
        mar= smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        #print(shape)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        
        



        if mar <= .3 or mar > .38 : 
            counter = counter + 1
        else:
            if counter >= 15:
                total += 1
                frame = vs.read()

                cv2.putText(frame, "3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                
                frame2= frame.copy()
                
                img_name = "picture_{}.png".format(total)
                
                cv2.imwrite(img_name, frame)

                
                print("{} written!".format(img_name))

            counter = 0


    cv2.imshow("Frame", frame)
    fps.update()

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break

fps.stop()


cv2.destroyAllWindows()
vs.stop()