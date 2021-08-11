import cv2
import numpy as np
import math
import pytesseract
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# src = cv2.imread('C:\_Project\OCR\Sample\doc8.jpg')
src = cv2.imread('C:\_Project\OCR\Sample\doc8.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('C:\_Project\OCR\Sample\doc8.jpg')
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# t, t_otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
at = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

canny = cv2.Canny(at, 5000, 1500, apertureSize = 5, L2gradient = True)
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength = 100, maxLineGap = 500)
# lines = cv2.HoughLines(canny, 0.8, np.pi / 180, 150, srn = 100, stn = 100, min_theta = 0, max_theta = np.pi)

print('검출 라인 개수 : {:d}'.format(len(lines)))

totalAngle = 0.0
totalAngleCount = 0.0
angleList = {}

for i in lines:
    x1 = i[0][0]
    y1 = i[0][1]
    x2 = i[0][2]
    y2 = i[0][3]

    x = x1 - x2
    y = y1 - y2
    if x < 0:
        x = -x
    if y < 0:
        y = -y

    angle = 0.0

    if x > y :
        angle = round(math.degrees(math.atan(y/x)), 2)
    else :
        angle = round(math.degrees(math.atan(x/y)), 2)

    if angle != 0 :
        if angle in angleList :
            angleList[angle] += 1
        else :
            angleList[angle] = 1

        if angle < 2.5:
            totalAngle += angle
            totalAngleCount += 1.0
            cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
            cv2.circle(dst, (x1, y1), 3, (255, 0, 0), 5, cv2.FILLED)
            cv2.circle(dst, (x2, y2), 3, (255, 0, 0), 5, cv2.FILLED)

if totalAngleCount > 0 :
    averageAngle = totalAngle / totalAngleCount
else :
    averageAngle = 0

maxAngle = max(angleList, key=angleList.get)

sortedList = sorted(angleList.items(), key=operator.itemgetter(1), reverse=True)

size = 0
totalCount = 0
totalAngle = 0.0
for ang in sortedList :
    size += 1
    if size <= 3 :
        angle = ang[0]
        angCount = ang[1]
        totalCount += angCount
        totalAngle += (angle * angCount)
    else :
        break

maxAverageAngle = round(totalAngle / totalCount, 2)

print('averageAngle : {:.2f}'.format(averageAngle))
print('maxAngle : {:.2f}'.format(maxAngle))
print('maxAverageAngle (3) : {:.2f}'.format(maxAverageAngle))

if totalAngleCount > 5 :
    height, width = src.shape

    matrix = cv2.getRotationMatrix2D((width/2, height/2), maxAverageAngle, 1)
    rotated = cv2.warpAffine(src, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
else :
    rotated = src.copy()

# rotated_threshold = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)
rotated_threshold = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# print(rotated_threshold)
# print(len(rotated_threshold))
# print(len(rotated_threshold[0]))
# print(len(rotated_threshold[1]))

# rowMax = list(range(0, len(rotated_threshold[0]), 0))
# columnMax = list(range(0, len(rotated_threshold), 0))

# np.savetxt('file01.txt', rt, fmt='%d', delimiter=' ')
rt = rotated_threshold.copy()

cv2.circle(rotated, (0, 0), 3, (255, 0, 0), 5, cv2.FILLED)
cv2.imshow('rotated', rotated)

rowLength = len(rt)
colLength = len(rt[0])

print('Length row / col : ', rowLength, ' / ', colLength)

rowMax = [0] * rowLength
colMax = [0] * colLength

rowIndex = 0
for row in rt :
    colIndex = 0
    for column in row :
        if column == 0 :
            colMax[colIndex] += 1
            rowMax[rowIndex] += 1
        colIndex += 1
    rowIndex += 1

rowMaxAvg = np.mean(rowMax)
rowMaxHalf = max(rowMax) / 2
print('rowMaxMax : ', max(rowMax))
print('rowMaxAvg : ', rowMaxAvg)

rowMaxHalfCount = 0
# rowMaxAvgCount = 0
index = 0
for row in rowMax :
    if row >= rowMaxHalf :
        rowMaxHalfCount += 1
    else :
        rowMax[index] = 0
    index += 1
    # if row > rowMaxAvg :
    #     rowMaxAvgCount += 1

print('rowMaxHalfCount : ', rowMaxHalfCount)
# print('rowMaxAvgCount : ', rowMaxAvgCount)

colMaxAvg = np.mean(colMax)
colMaxHalf = max(colMax) / 2
print('colMaxMax : ', max(colMax))
print('colMaxAvg : ', colMaxAvg)

colMaxHalfCount = 0
# colMaxAvgCount = 0
index = 0
for col in colMax :
    if col >= colMaxHalf :
        colMaxHalfCount += 1
    else :
        colMax[index] = 0
    index += 1
    # if col > colMaxAvg :
    #     colMaxAvgCount += 1

print('colMaxHalfCount : ', colMaxHalfCount)
# print('colMaxAvgCount : ', colMaxAvgCount)

rowX = range(0, len(rowMax))
colX = range(0, len(colMax))

plt.subplot(211)
plt.plot(rowX, rowMax)
plt.title('row')
plt.subplot(212)
plt.plot(colX, colMax)
plt.title('column')
plt.show()



# pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'
#
# str = pytesseract.image_to_string(rotated, lang='kor')
# print(str)
# cv2.imshow("origin", src)
# cv2.imshow("canny", canny)
# cv2.imshow("threshold", at)
# cv2.imshow("line", dst)
# cv2.imshow('rotated', rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
