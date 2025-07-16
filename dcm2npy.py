import SimpleITK as sitk
import numpy as np
import os

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("./dcm/20240201000938/201")
reader.SetFileNames(dicom_names)
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image)  # z, y, x
origin = image.GetOrigin()  # x, y, z
spacing = image.GetSpacing()  # x, y, z
outputpath = "./npy/volume.npy"
np.save(outputpath, image_array)