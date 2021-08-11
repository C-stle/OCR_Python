import cv2
import numpy as np
import os
import matplotlib.pylab as plt

# path = os.path.join('C:\_Project\OCR\Sample', 'document1.jpg');
# src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# t, t_otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# at = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

# cv2.imshow('src', src)
# cv2.imshow('otsu', t_otsu)
# cv2.imshow('at', at)

# cv2.waitKey()
# cv2.destoryAllWindows()

# image = cv2.imread('C:\_Project\OCR\Sample\document1.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = 255 - gray
#
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# coords = np.column_stack(np.where(thresh > 0))
#
# angle = cv2.minAreaRect(coords)[-1]
#
# print(angle)
# if angle < -45:
#     angle = 90 + angle
# else :
#     angle = -angle
#
# print(angle)
#
# (h, w) = image.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
# cv2.imshow('thres', thresh)
# cv2.imshow('rotated', rotated)
#
# cv2.waitKey()
# src = cv2.imread('C:\_Project\OCR\Sample\document1.jpg', cv2.IMREAD_GRAYSCALE)

src = cv2.imread('C:\_Project\OCR\Sample\document3.jpg')

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('thresh : ')
print(thresh)
thresh = thresh[1]

# np.where([조건문]) 조건에 해당하는 인덱스 출력

coords = np.column_stack(np.where(thresh > 0))
print('coords : ')
print(coords)
angle = cv2.minAreaRect(coords)[-1]
print(angle)
if angle < -45 :
    angle = -(90 + angle)
else :
    angle = -angle

height, width = gray.shape
# angle = 0;


matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

# matrix = cv2.getRotationMatrix2D((width/2, height/2), 1, 1)
rotated = cv2.warpAffine(src, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print("[INFO] angle: {:.3f}".format(angle))

cv2.imshow('origin', src)
# cv2.imshow('threshold', at)
cv2.imshow('rotated', rotated)

cv2.waitKey(0)
cv2.destoryAllWindows()
