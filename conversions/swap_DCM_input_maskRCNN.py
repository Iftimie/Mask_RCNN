import dicom
import os
import numpy as np
import cv2
from natsort import natsorted
from nifti import *
import scipy.ndimage as ndimage

PathDicom = "../../rocketChallenge_data/smir/"
lstFilesNII = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii" in filename.lower():  # check whether the file's DICOM
            lstFilesNII.append(os.path.join(dirName,filename))

lstFilesNII = natsorted(lstFilesNII)

for i in range(len(lstFilesNII)):
    nim = NiftiImage(lstFilesNII[i])
    data = nim.data
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
    data = np.transpose(data,(0,2,1))
    data = (np.array(data,dtype=np.float32) / data.max() * 255).astype(np.uint32)
    data = ndimage.zoom(data,(0.5,0.5,0.5))
    print data.shape
    nim = NiftiImage(data)
    nim.save("../../rocketChallenge_data/smir/input_MaskRCNN/MRI_"+str(i+1)+".nii")

    # for x in range(data.shape[0]):
    #     piece = np.array(data[x,:,:],dtype=np.float32)/ data.max()
    #     cv2.imshow("slice",piece)
    #     cv2.waitKey(10)

