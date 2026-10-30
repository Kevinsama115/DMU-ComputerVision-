#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:02:07 2026

@author: kevin
"""

import cv2 as cv 
import matplotlib.pyplot as plt 
 
img = cv.imread('/Users/kevin/Desktop/mistyroad.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
plt.imshow(gray,cmap='gray') , plt.xticks([]) , plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(h,color='r',linewidth=1),plt.show()

equal=cv.equalizeHist(gray)
plt.imshow(equal,cmap='gray'),plt.xticks([]), plt.ysticks([]),plt.show()

h=cv.calHist([equal],[0],None,[256],[0,256],) 
plt.plot(h,color='r',linewidth=1),plt.show()
 