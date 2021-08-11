import cv2
import pytesseract
import os
import numpy as np

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'
cell_str = ''

path_dir = 'C:/_Project/OCR/Vision/OutPut/Cell'

file_list = os.listdir(path_dir)

print(file_list)

for file in file_list :
    img = cv2.imread(path_dir + '/' + file)
    cell_str += pytesseract.image_to_string(img, lang='kor')
# img = cv2.imread('C:\_Project\OCR\Sample\document3.jpg')
# img = cv2.imread('C:\_Project\OCR\Vision\OutPut\Cell\Cell_1_5.bmp')

file = open('cell_str.txt', 'w', encoding='utf8')
file.truncate(0)
file.write(cell_str)
file.close()
print(cell_str)

img = cv2.imread('C:\_Project\OCR\Sample\doc3.jpg')
doc_str = pytesseract.image_to_string(img, lang='kor')
file = open('doc_str.txt', 'w', encoding='utf8')
file.truncate(0)
file.write(doc_str)
file.close()
print(doc_str)

# custom_config = r'-l ko --oem 3 --psm 6'
# pytesseract.image_to_string(img, config=custom_config)
# str = pytesseract.image_to_string(img, lang='kor')
# print(str)

# cv2.imshow('img', img)
# cv2.waitKey(0)
