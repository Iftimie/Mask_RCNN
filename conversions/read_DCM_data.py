import dicom
import os
import numpy as np
import cv2
from natsort import natsorted
from nifti import *
import scipy.ndimage as ndimage

PathDicom = "../../rocketChallenge_data/smir/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    image_list = []
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            image_list.append(os.path.join(dirName,filename))
    lstFilesDCM.append(natsorted(image_list))

for x in range(1,len(lstFilesDCM)):
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[x][0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM[x]))
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    print ConstPixelDims
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    for slice in lstFilesDCM[x]:
        ds = dicom.read_file(slice)
        ArrayDicom[:, :, lstFilesDCM[x].index(slice)] = ds.pixel_array

    max =  ArrayDicom.max()
    ArrayDicom = np.array(ArrayDicom,dtype=np.float32) / max * 255

    newArrayDicom = np.zeros((len(lstFilesDCM[x]), int(RefDs.Rows), int(RefDs.Columns)), dtype=np.float32)
    height_idx = len(lstFilesDCM[x]) -1
    for i in range(len(lstFilesDCM[x])):
        newArrayDicom[height_idx - i,:,:] = ArrayDicom[:,:,i] # now newArrayDicom has the veritical axis from bottom to top (as in original nii)

    ArrayDicom = newArrayDicom
    newArrayDicom = np.zeros((len(lstFilesDCM[x]), int(RefDs.Rows), int(RefDs.Columns)), dtype=np.float32)
    width_idx = int(RefDs.Rows) -1
    for i in range(int(RefDs.Rows)):
        newArrayDicom[:,i,:] = ArrayDicom[:,width_idx-i,:]

    ArrayDicom = None

    #newArrayDicom = ndimage.zoom(newArrayDicom,(0.5,0.5,0.5))
    if len(lstFilesDCM[x]) > 1024:
        divider = 4
    else:
        divider = 2
    newArrayDicom_resized = np.zeros((len(lstFilesDCM[x])/divider, int(RefDs.Rows)/2, int(RefDs.Columns)/2), dtype=np.float32)
    for i in range(newArrayDicom_resized.shape[0]):
        newArrayDicom_resized[i,:,:] = cv2.resize(newArrayDicom[i*divider,:,:],(0,0),fx=0.5,fy=0.5)

    newArrayDicom = newArrayDicom_resized
    newArrayDicom_resized = np.zeros((256,256,256), dtype=np.float32)
    for i in range(newArrayDicom_resized.shape[0]):
        newArrayDicom_resized[i,:,:] = newArrayDicom[i,:,:]

    newArrayDicom = newArrayDicom_resized
    # print newArrayDicom.shape
    # for slice in range(newArrayDicom.shape[0]):
    #     data = np.array(newArrayDicom[slice,:,:],dtype=np.uint8)
    #     cv2.imshow("slice",data)
    #     cv2.waitKey(10)

    nim = NiftiImage(newArrayDicom)
    nim.save("../../rocketChallenge_data/smir/MRI_"+str(x)+".nii")


