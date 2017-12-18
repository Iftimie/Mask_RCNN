from nifti import *
import numpy as np
import cv2
from numpy import genfromtxt

nim = NiftiImage('MRI_orig_padded0_input_maskRCNN.nii')
nii_data = nim.data

import pandas as pd
df=pd.read_csv('out.csv', sep=',')
data = df.as_matrix()
y1 = int(data[0,2])
x1 = int(data[0,3])
z1 = int(data[0,4])
y2 = int(data[0,5])
x2 = int(data[0,6])
z2 = int(data[0,7])
print y1,x1,z1,y2,x2,z2

front_side = nii_data[:,:,z1]
back_side = nii_data[:,:,z2]
front_side = np.array(front_side,dtype=np.float32)/ nii_data.max()
back_side = np.array(back_side,dtype=np.float32)/ nii_data.max()
front_side = cv2.rectangle(front_side,(x1,y1),(x2,y2),(255),2)
back_side = cv2.rectangle(back_side,(x1,y1),(x2,y2),(255),2)
cv2.imshow("front side",front_side)
cv2.imshow("back side", back_side)
cv2.waitKey(100)

left_side = nii_data[:,x1,:]
right_side = nii_data[:,x2,:]
left_side = np.array(left_side,dtype=np.float32)/ nii_data.max()
right_side = np.array(right_side,dtype=np.float32)/ nii_data.max()
left_side = cv2.rectangle(left_side,(z1,y1),(z2,y2),(255),2)
right_side = cv2.rectangle(right_side,(z1,y1),(z2,y2),(255),2)
cv2.imshow("left_side",left_side)
cv2.imshow("right_side", right_side)
cv2.waitKey(100)

top_side = nii_data[y1,:,:]
down_side = nii_data[y2,:,:]
top_side=  np.array(top_side,dtype=np.float32)/ nii_data.max()
down_side = np.array(down_side,dtype=np.float32) / nii_data.max()
top_side = cv2.rectangle(top_side, (z1,x1),(z2,x2),(255),2)
down_side = cv2.rectangle(down_side, (z1,x1),(z2,x2),(255),2)
cv2.imshow("top_side",top_side)
cv2.imshow("down_side", down_side)
cv2.waitKey(100000)

