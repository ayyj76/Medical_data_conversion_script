import SimpleITK as sitk
import numpy as np
import cv2
import os

# -------------------- configuration --------------------
count     = 1
dcm_path  = "./dcm/20240201000938/201"
outputDir = "./jpg"                       # where the JPGs will be saved
# -------------------------------------------------------

# create output directory if it does not exist
os.makedirs(outputDir, exist_ok=True)

# sort the DICOM file names by the first numeric part (descending)
files = sorted(os.listdir(dcm_path),
               key=lambda x: int(x.split('_')[0]),
               reverse=True)

# ---------- helper: convert one DICOM slice to JPG ----------
def dicom_slice_to_jpg(img, low_window, high_window, save_path):
    """Apply windowing and save the slice as a high-quality JPEG."""
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg  = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # normalize
    newimg  = (newimg * 255).astype(np.uint8)
    cv2.imwrite(save_path, newimg,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# -----------------------------------------------------------

# main conversion loop
for fname in files:
    # full path to the DICOM file
    dcm_file = os.path.join(dcm_path, fname)

    # output JPG file name
    jpg_name = f"{count}.jpg"
    jpg_path = os.path.join(outputDir, jpg_name)

    # read the DICOM volume
    ds_array  = sitk.ReadImage(dcm_file)
    img_array = sitk.GetArrayFromImage(ds_array)

    # SimpleITK returns (Z,Y,X); keep the first slice (Z=0)
    img_array = img_array[0, :, :]          # shape: (Y, X)

    # compute min/max for windowing
    low_val  = np.min(img_array)
    high_val = np.max(img_array)

    # convert and save
    dicom_slice_to_jpg(img_array, low_val, high_val, jpg_path)
    print(f"Saved {jpg_path}")

    count += 1