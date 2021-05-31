import numpy as np
import imutils
import cv2
import pyzbar.pyzbar as pyzbar
import roslibpy
import time

## Method to detect boxes and BC bounds
def detectBCBound(frame):
    ## convert frame to grayscale:
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## apply median blurring to reduce salt-and-pepper noise:
    grayFrame = cv2.medianBlur(grayFrame, ksize=9)  # ksize is +ve odd int
    ## apply gaussian blurring to reduce Gaussian noise
    grayFrame = cv2.GaussianBlur(grayFrame, ksize=(9, 9), sigmaX=0, sigmaY=0)

    ## compute Scharr derivative magnitude representation of the
    # images in both x and y direction (more accurate than Sobel):
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Scharr(grayFrame, ddepth=ddepth, dx=1, dy=0)
    gradY = cv2.Scharr(grayFrame, ddepth=ddepth, dx=0, dy=1)

    ## subtract the y-gradient from the x-gradient of the Scharr operator
    # to give regions of high horiz gradients and low vertical gradients
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    (_, thresh) = cv2.threshold(gradient, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    eroded = cv2.dilate(thresh, erosionKernel, iterations=1)
    openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    opened = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, openingKernel, iterations=2)

    ## find contours in threshold image, then sort by area
    contours, hierarchy = cv2.findContours(opened.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # findContours (cv2) will modify the source frame, so copy.
    barcodeBoxes = []
    for i, contour in enumerate(contours):
        # if contour has children
        if hierarchy[0][i][2] != -1:
            # cv2.drawContours(frame,[contour],0,(0,255,0),3)
            epsilon = (10 ** -3) * cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            # cv2.drawContours(frame, [approx], 0, (0,255,255), 3)
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            scale = 10 ** 5
            if box is not None:
                if cv2.contourArea(box) < 4 * scale and cv2.contourArea(box) > scale:
                    cv2.drawContours(frame, [box], 0, (255, 0, 255), 3)
                if cv2.contourArea(box) < scale and cv2.contourArea(box) > 0.1 * scale:
                    barcodeBoxes.append(box)
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    success = False if len(barcodeBoxes) == 0 else True
    return success, barcodeBoxes

## Method to scan barcodes:
def scanBarcodes(frame, boxes, isNotEmpty):
    barcodeSet = set()
    ## pre-process frame (optimise?)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ''' need to figure out best threshold method...'''
    (_, thresh) = cv2.threshold(grayFrame, 100, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    if isNotEmpty == True:
        for box in boxes:
            ## get scan area (larger than detected position)
            min = np.min(box, axis=0)  # find top-left corner coords of bounding box
            max = np.max(box, axis=0)  # find bot-right corner coords of bounding box
            pad = 10
            roi = thresh[min[1] - pad: max[1] + pad, min[0] - pad: max[0] + pad]
            if roi.any() != 0:
                #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
                barcodes = pyzbar.decode(roi)
                for barcode in barcodes:
                    barcodeData = barcode.data.decode("utf-8")
                    barcodeType = barcode.type
                    # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
                    barcodeSet.add((barcodeType, barcodeData))
                    ## display barcode id on top LHS of box
                    idStr = "id: " + str(barcodeData)
                    cv2.putText(frame, idStr, (min[0], min[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #(x, y, w, h) = barcode.rect
                    #cv2.rectangle(roi, (x,y), (x+w,y+h), (0,0,255), 2)
                    #cv2.imshow("roi", roi)
    else:
        print("no barcode boxes to read!")
    return list(barcodeSet)

## Read video from file:
cap = cv2.VideoCapture('Test1.MOV')
print(cap.isOpened())
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

## Prep writer to save vid:
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20.0
capSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('detectBoxBarcode1.avi', fourcc, fps, capSize, isColor=1)
#outGray = cv2.VideoWriter('boundedBarcode2Blobs.avi', fourcc, fps, capSize, isColor=0)

'''
# in order to use this library, ROS envi needs to be setup to run rosBridge (ROS bridge suite)
## connect to ros bridge node on local host
rosClient = roslibpy.Ros(host="localhost", port=9090)
rosClient.run()
barcodeTalker = roslibpy.Topic(rosClient, '/barcodes', 'std_msgs/String')
'''

while(cap.isOpened()):
    isFrameAvail, frame = cap.read()
    if isFrameAvail:
        isNotEmpty, barcodeBoxes = detectBCBound(frame)
        barcodes = scanBarcodes(frame, barcodeBoxes, isNotEmpty)

        ## display frame and write out to vid
        cv2.imshow('frame',frame)
        #out.write(frame)

        '''
        ## publishing out to ros topic:
        if rosClient.is_connected:
            barcodeTalker.publish(roslibpy.Message({'barcodes':str(barcodes)}))
            print('Sending message...')
            #time.sleep(1)
        '''

        ## early-termination keystroke
        if cv2.waitKey(1) & 0xFF == 27:
            print("quit")
            break
    else:
        print("isFrameAvail = " + str(isFrameAvail))
        break

cap.release()
out.release()
#outGray.release()
cv2.destroyAllWindows()