import numpy as np
import imutils
import cv2
import pyzbar.pyzbar as pyzbar
import roslibpy
import time

## Method to detect bounding boxes for barcodes:
def detectBCBound(frame):
    ## convert frame to grayscale:
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## apply median blurring to reduce salt-and-pepper noise:
    grayFrame = cv2.medianBlur(grayFrame, ksize=5)  # ksize is +ve odd int
    ## apply gaussian blurring to reduce Gaussian noise
    grayFrame = cv2.GaussianBlur(grayFrame, ksize=(5, 5), sigmaX=0, sigmaY=0)

    ## compute Scharr derivative magnitude representation of the
    # images in both x and y direction (more accurate than Sobel):
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Scharr(grayFrame, ddepth=ddepth, dx=1, dy=0)
    gradY = cv2.Scharr(grayFrame, ddepth=ddepth, dx=0, dy=1)

    ## subtract the y-gradient from the x-gradient of the Scharr operator
    # to give regions of high horiz gradients and low vertical gradients
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    ## blur and threshold the frame to further smooth out high-freq noise in the frame:
    blurred = cv2.GaussianBlur(gradient, ksize=(5, 5), sigmaX=0, sigmaY=0)
    blurred = cv2.medianBlur(blurred, ksize=5)
    (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ## initial closing (close gap between barcode regions to create blobs):
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel, iterations=2)

    ## remove unwanted blobs / edges via erosion:
    erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    eroded = cv2.erode(closed, erosionKernel, iterations=15)

    ## dilate the wanted blobs:
    dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15, 15))
    dilated = cv2.dilate(eroded, dilationKernel, iterations=5)

    ## find contours in threshold image, then sort by area
    contours = cv2.findContours(dilated.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_NONE)  # findContours will modify the source frame, so copy.
    contours = imutils.grab_contours(contours) # gets contours from return-vals
    cArr = sorted(contours, key=cv2.contourArea, reverse=True)
    boxes = []

    for contour in cArr:
        ## draw contour using minimum area (rotated rectangle)
        rect = cv2.minAreaRect(
            contour)  # returns a Box2D structure with: ( center (x,y), (width, height), angle of rotation )
        box = cv2.boxPoints(rect)  # get four corners of the rectangle
        box = np.intp(box)  # turns into integer used for indexing; this is the bounding box of the barcode
        if box is not None:
            cv2.drawContours(frame, [box], contourIdx=0, color=(0, 255, 0), thickness=3)
            boxes.append(box)

    ## display frame and write out to vid
    #cv2.imshow('frame', frame)
    success = False if len(boxes) == 0 else True
    return success, boxes

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
                    (x, y, w, h) = barcode.rect
                    idStr = "id: " + str(barcodeData)
                    cv2.putText(frame, idStr, (min[0], min[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #cv2.rectangle(roi, (x,y), (x+w,y+h), (0,0,255), 2)
                    #cv2.imshow("roi", roi)
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
out = cv2.VideoWriter('detectedBarcode1.avi', fourcc, fps, capSize, isColor=1)
#outGray = cv2.VideoWriter('detectedBarcode2Blobs.avi', fourcc, fps, capSize, isColor=0)

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
        isNotEmpty, boxes = detectBCBound(frame)
        barcodes = scanBarcodes(frame, boxes, isNotEmpty)
        cv2.imshow("frame",frame)
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
'''
barcodeTalker.unadvertise()
rosClient.terminate()
'''
