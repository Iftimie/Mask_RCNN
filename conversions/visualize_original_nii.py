from nifti import *
import numpy as np
import cv2
import scipy.ndimage as ndimage

nim = NiftiImage('MRI_orig.nii') #actually it is 256,256,160
data = nim.data

for i in range(data.shape[2]):
    piece = np.array(data[:,:,i],dtype=np.float32)/ data.max()
    cv2.imshow("slice",piece)
    cv2.waitKey(100)