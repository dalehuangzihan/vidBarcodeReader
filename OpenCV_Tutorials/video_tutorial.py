import cv2
import datetime

## Prepping the capture variables for reading input video
'''cap = cv2.VideoCapture(0) # (or '-1') to select default camera as video source'''
cap = cv2.VideoCapture('Test1.MOV') # to select file as video source
print(cap.isOpened()) # always check if the path vid parse was successful
# read some properties from the cap-ed frame:
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

## Prepping the writer variables for writing out processed video to directory
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # is the fourcc code for the video | or ('M', 'J', 'P' 'G')
fps = 20.0
capSize = (int(frameWidth), int(frameHeight)) # is the size of my source video
'''Set isColor=0 to save grayscale:'''
outGray = cv2.VideoWriter('testVidOutputGrayscale.avi', fourcc, fps, capSize, isColor = 0)
'''set isColor=1 to save BGR (is default option)'''
outBGR = cv2.VideoWriter('testVidOutputBGR.avi',fourcc,fps,capSize,isColor = 1)

## Capture video input from 'cap' frame by frame:
while(cap.isOpened()):
    isFrameAval, frame = cap.read() # isFrameAvail is a bool indicating whether the current frame is captured / avail

    if isFrameAval:
        print(frameWidth, frameHeight)

        ## print current date-time (and a rectangle) onto every frame:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(datetime.datetime.now())
        cv2.putText(frame,text,(10,50),font,fontScale=1,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.rectangle(frame,pt1=(100,100),pt2=(300,500),color=(0,255,0),thickness=4,lineType=cv2.LINE_AA)

        grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', grayscaleFrame)

        outBGR.write(frame)
        outGray.write(grayscaleFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # here we do 1ms display time per frame to keep up with the framerate
            print("quit")
            break
    else:
        break

cap.release() # REMEMBER TO RELEASE THE CAPTURE INSTANCE!
outGray.release() # REMEMBER TO RELEASE THE OUT INSTANCE!
cv2.destroyAllWindows()