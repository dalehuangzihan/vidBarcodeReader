import numpy as np
import cv2

#img = cv2.imread('lena.jpg',1)

## Draw image using numpy array (mtx of 0s gives a black image)
img = np.zeros([512,512,3], np.uint8)
                        # z-coord = 3 to indicate 3-channel image


'''
Top-left-hand corner of image has coordinate (0,0).
Bottom-right-hand corner of iamge has coordinates (imagHeight, imageWidth)
'''
## Drawing line and arrowed line:
linePt1 = (0,0)
linePt2 = (255,255)
lineColour = (0,255,0) # is in BGR channel format (reverse of RGB!)
lineThickness = 3
img = cv2.line(img, linePt1, linePt2, lineColour, lineThickness)
img = cv2.arrowedLine(img, (0,255), linePt2, (255,0,0), lineThickness)

## Drawing rectangle:
topLeftCoord = (300,50)
bottomRightCoord = (510, 128)
rectThickness = 5
rectLineType = cv2.LINE_4
img = cv2.rectangle(img, topLeftCoord, bottomRightCoord, lineColour, rectThickness, rectLineType)

## Drawing circle:
circleCenter = (300,400)
circleRadius = 70
circleColour = (0,0,255)
circleThickness = -1 # this will cause the circle to be shaded-in!
img = cv2.circle(img, circleCenter, circleRadius, circleColour, -1)

## Placing text:
font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 4
textColour = (255,255,255)
textThickness = 5
textLineType = cv2.LINE_AA
img = cv2.putText(img,'OpenCV!', (10,500), font, fontSize, textColour, textThickness, textLineType)


cv2.imshow('image',img)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
    print("quit")