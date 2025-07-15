import os
import SimpleITK as sitk
import numpy as np


def dcm2nii_sitk(path_read, path_save, output_name="data.nii.gz"):
    """
    Converts the DICOM series with the most images in the specified folder to a .nii.gz file.

    :param path_read: Path to the DICOM folder
    :param path_save: Path where the NIfTI file will be saved
    :param output_name: Name of the output file (e.g., 'data.nii.gz')
    :return: Path to the saved NIfTI file, or None if no DICOM series was found
    """

    # Create a reader for DICOM series
    reader = sitk.ImageSeriesReader()

    # Get all available DICOM series IDs in the directory
    series_ids = reader.GetGDCMSeriesIDs(path_read)

    if not series_ids:
        print(f"Warning: No DICOM series found in {path_read}")
        return None

    # Find the series with the maximum number of slices
    lengths = []
    for series_id in series_ids:
        dicom_files = reader.GetGDCMSeriesFileNames(path_read, series_id)
        lengths.append(len(dicom_files))

    max_index = np.argmax(lengths)  # Index of the series with the most slices
    dicom_files = reader.GetGDCMSeriesFileNames(path_read, series_ids[max_index])

    # Set the files to be read and execute the read operation
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # Ensure the save directory exists
    os.makedirs(path_save, exist_ok=True)

    # Construct the full output path
    output_path = os.path.join(path_save, output_name)

    # Write the image to disk
    sitk.WriteImage(image, output_path)
    print(f"Saved: {output_path}")

    return output_path


# Configuration settings
DICOM_PATH = r"./dcm/20240201000938/201"  # Path to a patient's DICOM folder
RESULT_PATH = r"./nii"  # Directory to save the NIfTI file
OUTPUT_NAME = "patient.nii.gz"  # Output filename(patient.nii is recommended,too)

# Run the conversion
dcm2nii_sitk(DICOM_PATH, RESULT_PATH, OUTPUT_NAME)
