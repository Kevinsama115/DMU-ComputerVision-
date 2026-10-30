#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:30:34 2026

@author: kevin
"""

import cv2 as cv 
img = cv.imread('/Users/kevin/Desktop/anime.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
 
canny1 = cv.Canny(gray,50,150) 
canny2 = cv.Canny(gray,100,200)  

cv.imshow('Orignal' , gray ) 
cv.imshow('Canny1' , canny1  ) 
cv.imshow('Canny2' , canny2  )

cv.waitKey()
cv.destoryAllWindows( )

 
