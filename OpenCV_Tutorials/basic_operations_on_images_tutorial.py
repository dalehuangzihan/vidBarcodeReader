import numpy as np
import cv2

img = cv2.imread('messi5.jpg') # is an image matrix
img3 = cv2.imread('opencv-logo.png') # is an image matrix

print(img.shape) # prints a touple of number of rows, columns, and channels
print(img.size) # prints the total number of pixels in the image
print(img.dtype) # prints Image datatype (encoding)

b,g,r = cv2.split(img) # splits image into their individual B,G,R channels (or any 3-channel encoding)
img2 = cv2.merge((b,g,r)) # merges the individual B,G,R channels back into a complete 3-channel image

## Use the xpos and ypos ranges for the ball to "frame" it up (which is our region of interest):
''' Takes the relevant image matrix cell values and stores it in a separate var (is itself a matrix).
Is essentially "cropping" a subset of the matrix and storing said subset (also a mtx) into a diff var.'''
ball = img[280:340, 330:390]
    # note: these coordinates can be obtained using mouse-actions and printouts
img2[273:333, 100:160] = ball
cv2.imshow('image2',img2)

## resize img2:
img2 = cv2.resize(img2,dsize=(512,512)) # dsize is destination size
## resize img3
img3 = cv2.resize(img3,dsize=(512,512))
## add two images (of the same size) together
img4 = cv2.add(img2,img3)
''' SUMS the matrix cell values of the two image matrices together (like superpostition) '''
''' size of the two image matrices must match! '''
cv2.imshow('image4',img4)

## add weighted images:
img5 = cv2.addWeighted(src1=img2,alpha=0.7,src2=img3,beta=0.3,gamma=0)
    # alpha is the weight of src1 mtx elems
    # beta is the weight of src2 mtx elems
    # gamma = scalar added to each sum

cv2.imshow('image5',img5)
cv2.waitKey(0)
cv2.destroyAllWindows()