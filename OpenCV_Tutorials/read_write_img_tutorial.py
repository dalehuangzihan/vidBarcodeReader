import cv2

## READING AND WRITING IMAGES ##
lenaImg = cv2.imread('lena.jpg', 0) # load image in grayscale
    # will return a matrix of pixel values if path is correct and read correctly

print(lenaImg)

cv2.imshow('this is the lenna image', lenaImg)
#cv2.waitKey(5000) # waits for number of millisec for which we want to show the image
    # waitKey(0) will wait for manual closing of window
k = cv2.waitKey(0) & 0xFF # AND with mask "0xFF" for 64-bit machines
'''
    0xFF is a hexadecimal constant which is 11111111 in binary. 
    By using bitwise AND ( & ) with this constant, it leaves only 
    the last 8 bits of the original (in this case, whatever cv2. waitKey(0) is).
'''

if k == 27: # 27 is the keystroke code for 'esc' key
    cv2.destroyAllWindows()
    print("escaped")
elif k == ord('s'): # ord returns an integer representing the Unicode character
    cv2.imwrite('lena_grayscale.png',lenaImg)
    print("image saved")
    cv2.destroyAllWindows()
