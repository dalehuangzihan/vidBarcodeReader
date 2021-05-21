import numpy as np
import cv2

## list  out all EVENTS available in cv2 library
events = [i for i in dir(cv2) if 'EVENT' in i] # dir() gives all classes and member function names inside cv2 package
#print(events)

''' Create the mouse-click event Callback function (fn head is standard) '''

def click_event(event, xpos, ypos, flags, param):
    ## listen for LBUTTONDOWN event:
    # for this event, we want to connect a line between two L-clicked points on the img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,center=(xpos,ypos),radius=5,color=(0,0,255),thickness=-1)
        clickedPoints.append((xpos,ypos))
        if len(clickedPoints) > 1:
            prevClickedPoint = clickedPoints[-1] # gives the last elem in the array
            prevprevClickedPoint = clickedPoints[-2] # gives the second-last elem in the array
            #draws line between prev and prevprev clicked points
            cv2.line(img,pt1=prevClickedPoint,pt2=prevprevClickedPoint,color=(0,0,255),thickness=3)
        cv2.imshow('image',img)

    ## listen for LBUTTONUP event:
    # for this event, we want to show the coordinates at the clicked point
    if event == cv2.EVENT_LBUTTONUP:
        print(xpos,', ',ypos)
        strXY = str(xpos) + ', ' + str(ypos)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,strXY,org=(xpos,ypos),fontFace=font,fontScale=.5,color=(0,255,0),thickness=2)
        cv2.imshow('image',img)

    ## listen for RBUTTONUP event:
    # for this event, we want to show the BGR channel info at the clicked point, and display colour in a separate window
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[ypos,xpos,0] # in BGR format, blue channel is at indx 0
        green = img[ypos,xpos,1]
        red = img[ypos,xpos,2]
        strBGR = str(blue) + ', ' + str(green) + ', ' + str(red) # print out the BGR colour channel values
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, strBGR, org=(xpos, ypos), fontFace=font, fontScale=.5, color=(0, 255, 0), thickness=2)
        cv2.imshow('image', img)

        # prepping colour-display window:
        myColourImage = np.zeros((512,512,3),np.uint8)
        myColourImage[:] = [blue,green,red] # copies 3-channel BGR info into the 3 z-coord cells for all x,y coords in myColorImage
        cv2.putText(myColourImage,strBGR,org=(10,50),fontFace=font,fontScale=2,color=(0,255,0),thickness=2)
        cv2.imshow('colour',myColourImage)

#img = np.zeros((512,512,3),np.uint8)
img = cv2.imread('lena.jpg')
cv2.imshow('image',img)
clickedPoints = []

''' Call our callback method whenever someone triggers a mouse event on the image '''
cv2.setMouseCallback(window_name='image',on_mouse=click_event)

k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

