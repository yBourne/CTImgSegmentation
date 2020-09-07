#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

img_part = cv2.imread('preprocessing.png')
img_part_ = cv2.medianBlur(img_part, 3)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("img", )
cv2.imshow("img", img_part_) #show mark
k = cv2.waitKey(0)
if k ==27:
    cv2.destroyAllWindows() 
# gray = cv2.cvtColor(img_part_, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

