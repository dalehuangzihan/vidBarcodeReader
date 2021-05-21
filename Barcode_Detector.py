import numpy as np
#import argparse # for parsing command-line arguments
import imutils
import cv2

## Read video from file:
cap = cv2.VideoCapture('Test2.MOV')
print(cap.isOpened())
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

## Prep writer to save vid:
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20.0
capSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('boundedBarcode2.avi', fourcc, fps, capSize, isColor=1)
outGray = cv2.VideoWriter('boundedBarcode2Blobs.avi', fourcc, fps, capSize, isColor=0)

while(cap.isOpened()):
    isFrameAvail, frame = cap.read()
    if isFrameAvail:
        ## convert frame to grayscale:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## apply median blurring to reduce salt-and-pepper noise:
        grayFrame = cv2.medianBlur(grayFrame, ksize=5) # ksize is +ve odd int
        ## apply gaussian blurring to reduce Gaussian noise
        grayFrame = cv2.GaussianBlur(grayFrame,ksize=(5,5),sigmaX=0,sigmaY=0)

        ## compute Scharr derivative magnitude representation of the
        # images in both x and y direction (more accurate than Sobel):
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Scharr(grayFrame, ddepth=ddepth, dx=1, dy=0)
        gradY = cv2.Scharr(grayFrame, ddepth=ddepth, dx=0, dy=1)

        ## subtract the y-gradient from the x-gradient of the Scharr operator
        # to give regions of high horiz gradients and low vertical gradients
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        #cv2.imshow('gradient', gradient)

        ## blur and threshold the frame to further smooth out high-freq noise in the frame:
        blurred = cv2.GaussianBlur(gradient,ksize=(5,5),sigmaX=0,sigmaY=0)
        blurred = cv2.medianBlur(blurred,ksize=5)
        #cv2.imshow('blurred', blurred)
        (_, thresh) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('thresh', thresh)

        ## initial closing (close gap between barcode regions to create blobs):
        closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel, iterations=2)
        #cv2.imshow('closed',closed)

        ## remove unwanted blobs / edges via erosion:
        erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5,5))
        eroded = cv2.erode(closed, erosionKernel, iterations=15)
        #cv2.imshow('eroded',eroded)

        ## dilate the wanted blobs:
        dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15,15))
        dilated = cv2.dilate(eroded,dilationKernel,iterations=5)

        cv2.imshow('testFrame', dilated)
        outGray.write(dilated)

        ## find contours in threshold image, then sort by area
        contours = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # findContours will modify the source frame, so copy.
        contours = imutils.grab_contours(contours)
        cArr = sorted(contours, key = cv2.contourArea, reverse=True)

        for contour in cArr:
            ## draw contour using minimum area (rotated rectangle)
            rect = cv2.minAreaRect(contour) # returns a Box2D structure with: ( center (x,y), (width, height), angle of rotation )
            box = cv2.boxPoints(rect) # get four corners of the rectangle
            box = np.intp(box) # turns into integer used for indexing
            cv2.drawContours(frame,[box],contourIdx=0,color=(0,255,0),thickness=3)

        ## display frame and write out to vid
        cv2.imshow('frame',frame)
        out.write(frame)

        ## early-termination keystroke
        if cv2.waitKey(1) & 0xFF == 27:
            print("quit")
            break
    else:
        print("isFrameAvail = " + str(isFrameAvail))
        break

cap.release()
out.release()
outGray.release()
cv2.destroyAllWindows()