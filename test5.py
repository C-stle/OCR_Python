import cv2
import numpy as np
import math
import pytesseract
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

src = cv2.imread('C:\_Project\OCR\Sample\doc3.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('C:\_Project\OCR\Sample\doc3.jpg')
cv2.imshow('origin', dst)
at = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

# canny = cv2.Canny(at, 5000, 1500, apertureSize = 5, L2gradient = True)
canny = cv2.Canny(src, 5000, 1500, apertureSize = 5, L2gradient = True)
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength = 200, maxLineGap = 1000)
# lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 10, maxLineGap = 100)

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
        angle = math.degrees(math.atan(y/x))
    else :
        angle = math.degrees(math.atan(x/y))

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

cv2.imshow('dst', dst)

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

    # matrix = cv2.getRotationMatrix2D((width/2, height/2), maxAverageAngle, 1)
    matrix = cv2.getRotationMatrix2D((width/2, height/2), maxAngle, 1)
    rotated = cv2.warpAffine(src, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print('rotated')
else :
    rotated = src.copy()

rotated_threshold = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

rt = rotated_threshold.copy()

rowLength = len(rt[0])
colLength = len(rt)

print('Length row / col : ', rowLength, ' / ', colLength)

rowCheckRange = rowLength - 10
colCheckRange = colLength - 10

defaultCheckCount = 5

rowLine = []
# 가로 체크만 해보자
x = 5
y = 5

while x < rowCheckRange and y < colCheckRange :
    if rt[y][x] == 0 :
        rowLine.append([x, y, 0, 0])
        checkCount = defaultCheckCount
        index = 1
        x += 1
        lineLength = 0

        while checkCount > 0 and x < rowCheckRange :
            if rt[y][x] == 0 :
                checkCount = defaultCheckCount
                rowLine[-1][2] = x
                rowLine[-1][3] = y
                lineLength += 1
            else :
                checkCount -= 1

            x += 1
        if lineLength < 100 :
            rowLine.pop()   # 일정 길이보다 작은 경우, 라인 삭제

    else :
        x += 1

    # 다음 column 로 이동
    if x >= rowCheckRange :
        y += 1
        x = 5

print('rowLine length : ', len(rowLine))
print(rowLine)

colLine = []
# 세로 체크만 해보자
x = 5
y = 5

while x < rowCheckRange and y < colCheckRange :
    if rt[y][x] == 0 :
        colLine.append([x, y, 0, 0])
        checkCount = defaultCheckCount
        index = 1
        y += 1
        lineLength = 0
        while checkCount > 0 and y < colCheckRange :
            if rt[y][x] == 0 :
                checkCount = defaultCheckCount
                colLine[-1][2] = x
                colLine[-1][3] = y
                lineLength += 1
            else :
                checkCount -= 1
            y += 1
        if lineLength < 50 :
            colLine.pop()   # 일정 길이보다 작은 경우, 라인 삭제
    else :
        y += 1

    # 다음 column 로 이동
    if y >= colCheckRange :
        x += 1
        y = 5

print('colLine length : ', len(colLine))
print(colLine)

white_img = np.full((colLength, rowLength), 255, dtype=np.uint8)

# row, 가로 줄
minRowX = rowLength
minRowY = colLength
maxRowX = 0
maxRowY = 0
for i in rowLine :
    cv2.line(white_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
    # cv2.circle(white_img, (i[0], i[1]), 2, (0, 0, 0), 2, cv2.FILLED)
    # cv2.circle(white_img, (i[2], i[3]), 2, (0, 255, 0), 2, cv2.FILLED)
    if minRowX > i[0] :
        minRowX = i[0]
    if maxRowX < i[0] :
        maxRowX = i[0]
    if minRowY > i[1] :
        minRowY = i[1]
    if maxRowY < i[1] :
        maxRowY = i[1]

    if minRowX > i[2] :
        minRowX = i[2]
    if maxRowX < i[2] :
        maxRowX = i[2]
    if minRowY > i[3] :
        minRowY = i[3]
    if maxRowY < i[3] :
        maxRowY = i[3]

# print('minRowX : ', minRowX)
# print('minRowY : ', minRowY)
# print('maxRowX : ', maxRowX)
# print('maxRowY : ', maxRowY)

# cv2.circle(white_img, (minRowX, minRowY), 3, (0, 255, 0), 5, cv2.FILLED)
# cv2.circle(white_img, (maxRowX, maxRowY), 3, (0, 255, 0), 5, cv2.FILLED)

# column, 세로 줄
minColX = rowLength
minColY = colLength
maxColX = 0
maxColY = 0
for i in colLine :
    cv2.line(white_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
    if minColX > i[0] :
        minColX = i[0]
    if maxColX < i[0] :
        maxColX = i[0]
    if minColY > i[1] :
        minColY = i[1]
    if maxColY < i[1] :
        maxColY = i[1]

    if minColX > i[2] :
        minColX = i[2]
    if maxColX < i[2] :
        maxColX = i[2]
    if minColY > i[3] :
        minColY = i[3]
    if maxColY < i[3] :
        maxColY = i[3]

# print('minColX : ', minColX)
# print('minColY : ', minColY)
# print('maxColX : ', maxColX)
# print('maxColY : ', maxColY)

# cv2.circle(white_img, (minColX, minColY), 3, (0, 255, 0), 5, cv2.FILLED)
# cv2.circle(white_img, (maxColX, maxColY), 3, (0, 255, 0), 5, cv2.FILLED)

minX = int((minRowX + minColX) / 2)
minY = int((minRowY + minColY) / 2)
maxX = int((maxRowX + maxColX) / 2)
maxY = int((maxRowY + maxColY) / 2)

print('minX : ', minX)
print('minY : ', minY)
print('maxX : ', maxX)
print('maxY : ', maxY)

cv2.circle(white_img, (minX, minY), 3, (0, 255, 0), 3, cv2.FILLED)
cv2.circle(white_img, (maxX, maxY), 3, (0, 255, 0), 5, cv2.FILLED)


# resize_rotated = cv2.resize(rotated_threshold, dsize=(717, 1000), interpolation=cv2.INTER_AREA)
# cv2.imshow('rotated_threshold', resize_rotated)
resize_white = cv2.resize(white_img, dsize=(717, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('white_lines', resize_white)

cv2.imshow('rotated', rotated)
# canny_rotated = cv2.Canny(rotated, 5000, 1500, apertureSize = 5, L2gradient = True)
# cv2.imshow('canny_rotated', canny_rotated)
# print(canny_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
