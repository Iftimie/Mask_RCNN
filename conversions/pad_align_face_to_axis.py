from nifti import *
import numpy as np
import cv2
import scipy.ndimage as ndimage

nim = NiftiImage('MRI_orig.nii') #actually it is 256,256,160
data = nim.data
print data.shape

new_data = np.zeros((256,256,160))

#Y view
for x in range(data.shape[0]):
    new_data[255-x,:,:] = data[x,:,:]
data = new_data
new_data = np.zeros((256,160,256))

#X view
for x in range(data.shape[1]):
    new_data[:,:,x] = data[:,x,:]
# NO NEED TO SET new_data[:,x,:] = data[:,:,x] after the loop above. it results in the same thing
# cv2.imshow("slice data",np.array(data[:,:,100],dtype=np.float32)/ data.max())
# cv2.imshow("slice new_data",np.array(new_data[:,100,:],dtype=np.float32)/ data.max())
# cv2.waitKey(10000)
#Z view
data = new_data
new_data = np.zeros((256,160,256))
for x in range(data.shape[2]):
    new_data[:,:,255-x] = data[:,:,x]
data = new_data

#Pad the image with 0 on the X axis
new_data = np.zeros((256,256,256))
for x in range(data.shape[1]):
    new_data[:,43+x,:] = data[:,x,:]

data = new_data
for x in range(data.shape[2]):
    piece = np.array(data[:,:,x],dtype=np.float32)/ data.max()
    cv2.imshow("slice",piece)
    cv2.waitKey(1)

data = data.astype(np.int32)
print data.shape
nim = NiftiImage(data)
nim.save("MRI_orig_padded0.nii")
