import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

src = cv2.imread('C:\_Project\OCR\Sample\document3.jpg')

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
print(thresh)

np.savetxt('file01.txt', thresh, fmt='%d', delimiter=' ')
