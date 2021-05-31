import cv2
from pyzbar import pyzbar
import numpy as np

cap = cv2.VideoCapture('Test1.MOV')

while(cap.isOpened()):
    isFrameAvail, frame = cap.read()
    if isFrameAvail:
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            data = barcode.data.decode('utf-8')
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (0,255,0), 5)



        cv2.imshow('Result',frame)
        cv2.waitKey(1)

