from nifti import *
import numpy as np
import cv2
import scipy.ndimage as ndimage

nim = NiftiImage('MRI_orig_resized.nii') #actually it is 160,256,256
data = nim.data
print nim.header['dim']
for x in range(data.shape[2]):
    piece = np.array(data[:,:,x],dtype=np.float32)/ data.max() # not image_data[:,:,x].max()
    piece_resized = cv2.resize(piece,(0,0),fx=0.5,fy=0.5)
    print piece_resized.shape
    if x %2==0:
        cv2.imshow("slice_resized",piece_resized)
        cv2.imshow("slice",piece)
        cv2.waitKey(100)

#resized_data = ndimage.zoom(data,(0.5,0.5,0.5)) #ndimage.zoom(data,0.5)

# for x in range(resized_data.shape[2]):
#     piece = np.array(resized_data[:,:,x],dtype=np.float32)/ resized_data.max()
#     print piece.shape
#     cv2.imshow("slice",piece)
#     cv2.waitKey(100)
#
# print 'ok'

# nim = NiftiImage(resized_data)
# print nim.header['dim']
# nim.save("MRI_orig_resized.nii")