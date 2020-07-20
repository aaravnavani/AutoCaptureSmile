# Introduction 

In this project, we use python and computer vision to auto capture a selfie by detecting if the user is smiling or not. 

## Importing 

We first have to import some libraries in order for our code to work: 

```python
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2
```

## Facial landmark detector

Inside the dlib package, there is an API called the facial landmark detector. This API has 68 (x,y) coordinates which point to specific facial structures. Here are all the 68 points visualized: 

![Facial Landmarks](faciallandmarks.jpg)


Since we want to detect the user's smile, we want to focus on the points in the range [49, 68]. We get these features by using the following piece of code: 

```python
shape_predictor= "shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
```

The first three lines load the .dat file and the last line extracts the coordinates that we want (the mouth in our case). You can download the .dat file from here: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

## Working with the smile 

Now we just have the 20 coordinates that represent the mouth: 

![Extracted Mouth](mouth.jpg)

In order to detect if the user is smiling or not, we create a mouth aspect ratio (MAR). In order to calculate this ratio, we have to find: 

- the distance between p49 and p55 
- the distance between p51 and p59
- the distance between p52 and p58
- the distance between p53 and p57 

Here is the formula to find the MAR: 

![MAR Equation](MARequation.jpg)

We can write a function in our code to perform to this calculation:

```python
def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar
```

We also want to detect the mouth from the user's face. In order to do this, we first convert the image to grayscale for simplicity. We run the predictor we loaded in earlier and get the MAR  value from the mouth detected. Finally, we draw a green outline around the mouth. 

```python
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
```

## Setting up the auto-capture 

With some experimentation, we see that we can "define" a smile to have a MAR value less than 0.3 or greater than 0.38. If it is, then we capture the frame and save the file with name "opencv_frame_<counter>.png"

```python
if mar <= .3 or mar > .38 : 
            counter = counter + 1
        else:
            if counter >= 15:
                total += 1
                frame = vs.read()

                
                
                frame2= frame.copy()
                
                img_name = "picture_{}.png".format(total)
                
                cv2.imwrite(img_name, frame)

                
                print("{} written!".format(img_name))

            counter = 0
 ```

Finally, we quit the video stream if the user presses "q" on their keyboard using the following code: 

```python
 key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
```

And that's it! 

## Running the code 

When we first run the code, the webcam loads up and a green circle outlines the mouth: 

*insert image*


When we smile, the program auto captures the frame and saves it in the same folder where the code is stored: 

When we open up our folder, we can see the image stored: 

* insert image* 









