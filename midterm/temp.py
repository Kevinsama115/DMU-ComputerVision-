# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv
import numpy as np

# 이미지 읽기 (np.ndarray)
img = cv.imread('soccer.jpg')
if img is None:
  raise FileNotFoundError('soccer.jpg파일을 찾을 수 없습니다.')

print('shape:', img.shape)  # 이미지 크기 (높이, 너비, 채널수 출력)
print('최솟값:', img.min())  # 전체 픽셀 중 최솟값(B,G,R 중 최소)
print('최댓값:', img.max())  # 전체 픽셀 중 최댓값
print('평균:', img.mean())  # 전체 픽셀 값 평균

print('최댓값 위치:', np.argmax(img))  # 최댓값의 1차원 인덱스
print('정렬후 앞부분:', np.sort(img, axis=None)[:10])  # 1차원으로 펴서 정렬 후 상위 10개

transposed = np.transpose(img, (1, 0, 2))  # 가로<->세로 변환
print('가로 세로 변환 후 모양:', transposed.shape)

cv.imshow('Original', img)  # 원본 이미지 창
cv.imshow('Transposed', transposed)  # 전치 이미지 창

cv.waitKey(0)  # 키 입력 대기
cv.destroyAllWindows()  # 창 닫기