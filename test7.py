import cv2
import numpy as np
import math
import pytesseract
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

path = 'C:\_Project\OCR\Sample\doc10.jpg'
thresholdBlockSize = 55
thresholdC = 5

src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
dst = cv2.imread(path)
resize_origin = cv2.resize(src, dsize=(707, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('origin', resize_origin)
at = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresholdBlockSize, thresholdC)
resize_threshold = cv2.resize(at, dsize=(707, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('threshold', resize_threshold)
# canny = cv2.Canny(at, 5000, 1500, apertureSize = 5, L2gradient = True)
canny = cv2.Canny(at, 5000, 1500, apertureSize = 5, L2gradient = True)
# lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength = 200, maxLineGap = 1000)
lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 100, minLineLength = 100, maxLineGap = 1000)

print('검출 라인 개수 : {:d}'.format(len(lines)))

totalAngle = 0
totalAngleCount = 0
zeroAngleCount = 0
angleList = {}

for i in lines:
    x1 = i[0][0]
    y1 = i[0][1]
    x2 = i[0][2]
    y2 = i[0][3]

    x = x2 - x1
    y = y2 - y1

    angle = 0.0

    if x == 0 or y == 0 :
        angle = 0
    elif x >= y :
        angle = math.degrees(math.atan(y/x))
    elif x < y :
        if y < 0 :
            angle = -(90 + math.degrees(math.atan(x/y)))
        else :
            angle = math.degrees(math.atan(x/y))

    # print('before angle : ', angle)
    if angle < -45:
        angle = 90 + angle
    # print('after angle : ', angle)

    # if angle != 0 :
    if angle < 5 and angle > -5:
        if angle in angleList :
            angleList[angle] += 1
        else :
            angleList[angle] = 1
        totalAngle += angle
        totalAngleCount += 1
        if angle == 0 :
            zeroAngleCount += 1
        else :
            cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
            cv2.circle(dst, (x1, y1), 3, (255, 0, 0), 5, cv2.FILLED)
            cv2.circle(dst, (x2, y2), 3, (255, 0, 0), 5, cv2.FILLED)

if totalAngleCount > 0 :
    averageAngle = totalAngle / totalAngleCount
else :
    averageAngle = 0
resize_dst = cv2.resize(dst, dsize=(707, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('dst', resize_dst)

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

if totalAngleCount > 0 :
    height, width = src.shape
    matrixAngle = 0
    if totalAngleCount - zeroAngleCount > zeroAngleCount :
        matrixAngle = maxAverageAngle
    else :
        print('set zero Angle')
    print('rotated Angle : ', matrixAngle)

    matrix = cv2.getRotationMatrix2D((width/2, height/2), matrixAngle, 1)
    rotated = cv2.warpAffine(src, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print('rotated')
else :
    rotated = src.copy()

rotated_threshold = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresholdBlockSize, thresholdC)

rt = rotated_threshold.copy()

rowLength = len(rt[0])
colLength = len(rt)

print('Length row / col : ', rowLength, ' / ', colLength)

exceptRange = 25

rowCheckRange = rowLength - exceptRange
colCheckRange = colLength - exceptRange

defaultCheckCount = 4

rowLine = []
# 가로 체크만 해보자
x = exceptRange
y = exceptRange

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
        x = exceptRange

print('rowLine length : ', len(rowLine))
# print(rowLine)

colLine = []
# 세로 체크만 해보자
x = exceptRange
y = exceptRange

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
        y = exceptRange

print('colLine length : ', len(colLine))
# print(colLine)

white_img = np.full((colLength, rowLength), 255, dtype=np.uint8)

# row, 가로 줄
minRowX = rowLength
minRowY = colLength
maxRowX = 0
maxRowY = 0

for i in rowLine :
    cv2.line(white_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)

    if minRowX + minRowY > i[0] + i[1] :
        minRowX = i[0]
        minRowY = i[1]

    if maxRowX + maxRowY < i[2] + i[3] :
        maxRowX = i[2]
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

    if minColX + minColY > i[0] + i[1]:
        minColX = i[0]
        minColY = i[1]

    if maxColX + maxColY < i[2] + i[3]:
        maxColX = i[2]
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

print('min X / Y : ', minX, ' / ', minY)
print('max X / Y : ', maxX, ' / ', maxY)

cv2.circle(white_img, (minX, minY), 3, (0, 255, 0), 3, cv2.FILLED)
cv2.circle(white_img, (maxX, maxY), 3, (0, 255, 0), 5, cv2.FILLED)


# resize_rotated = cv2.resize(rotated_threshold, dsize=(717, 1000), interpolation=cv2.INTER_AREA)
# cv2.imshow('rotated_threshold', resize_rotated)
resize_white = cv2.resize(white_img, dsize=(707, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('white_lines', resize_white)
resize_rotated = cv2.resize(rotated, dsize=(707, 1000), interpolation=cv2.INTER_AREA)
cv2.imshow('rotated', resize_rotated)
# canny_rotated = cv2.Canny(rotated, 5000, 1500, apertureSize = 5, L2gradient = True)
# cv2.imshow('canny_rotated', canny_rotated)
# print(canny_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
