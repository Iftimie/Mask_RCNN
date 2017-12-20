from nifti import *
import numpy as np
import cv2
import scipy.ndimage as ndimage

nim = NiftiImage('MRI_orig.nii') #actually it is 256,256,160
data = nim.data
print data.shape
#Pad the image with 0 on the X axis and save
new_data = np.zeros((256,256,256))
for x in range(data.shape[2]):
    new_data[:,:,43+x] = data[:,:,x]
data = new_data
data = data.astype(np.int32)
print data.shape
nim = NiftiImage(data)
nim.save("MRI_orig_padded0.nii")


new_data = np.zeros((256,256,256))
#flip Y axis
for x in range(data.shape[0]):
    new_data[255-x,:,:] = data[x,:,:]
data = new_data

new_data = np.zeros((256,256,256))
#flip Z axis
for x in range(data.shape[1]):
    new_data[:,255-x,:] = data[:,x,:]
data = new_data
#
# for x in range(data.shape[0]):
#     piece = np.array(data[x,:,:],dtype=np.float32)/ data.max()
#     cv2.imshow("slice",piece)
#     cv2.waitKey(10)

#swap the axes of z and y
data = np.transpose(data,(0,2,1))
data = (np.array(data,dtype=np.float32) / data.max() * 255).astype(np.uint32)
data = ndimage.zoom(data,(0.5,0.5,0.5))
print data.shape
nim = NiftiImage(data)
nim.save("MRI_orig_padded0_input_maskRCNN.nii")

for x in range(data.shape[0]):
    piece = np.array(data[x,:,:],dtype=np.float32)/ data.max()
    cv2.imshow("slice",piece)
    cv2.waitKey(10)



