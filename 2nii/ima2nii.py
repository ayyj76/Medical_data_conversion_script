import os
from typing import Optional

import SimpleITK as sitk


def dcm2nii_sitk(path_read: str,
                 path_save: str,
                 output_name: str = "data.nii.gz") -> Optional[str]:
    """
    Convert the DICOM/IMA series with the most slices in the given folder to .nii.gz
    :param path_read:  Directory containing the .IMA files
    :param path_save:  Directory where the .nii.gz file will be saved
    :param output_name: Output file name
    :return: Full path of the output file on success, otherwise None
    """

    # 1. Find all .IMA files in the directory
    ima_files = sorted(
        [os.path.join(path_read, f)
         for f in os.listdir(path_read)
         if f.lower().endswith(".ima")]
    )
    if not ima_files:
        print(f"Warning: No .IMA files found in {path_read}")
        return None

    # 2. Use ImageSeriesReader in file mode to read directly
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ima_files)

    # 3. Read and write
    image = reader.Execute()
    os.makedirs(path_save, exist_ok=True)
    out_path = os.path.join(path_save, output_name)
    sitk.WriteImage(image, out_path)
    print(f"Saved: {out_path}")
    return out_path


# -------------------- Configuration --------------------
DICOM_PATH  = r".\mayo\full_1mm\L067\full_1mm"  # Folder containing .IMA files
RESULT_PATH = r".\nii"
OUTPUT_NAME = "ima_patient.nii.gz"

dcm2nii_sitk(DICOM_PATH, RESULT_PATH, OUTPUT_NAME)
