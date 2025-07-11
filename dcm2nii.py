import os
import numpy as np
from pydicom.filereader import dcmread
import nibabel as nib
from nibabel.affines import from_matvec

def get_sorted_slices(dicom_folder):
    """Get DICOM slice files sorted by Z-position."""
    files = []
    for f in os.listdir(dicom_folder):
        if f.endswith('.dcm'):
            path = os.path.join(dicom_folder, f)
            try:
                ds = dcmread(path)
                pos = ds.ImagePositionPatient
                files.append((float(pos[2]), path))  # Sort by Z position
            except Exception as e:
                print(f"Skipping file {f}: {e}")
    files.sort(key=lambda x: x[0])  # Sort slices by Z-coordinate
    return [x[1] for x in files]

def get_pixel_spacing(ds):
    """Get pixel spacing in X and Y directions."""
    if hasattr(ds, 'PixelSpacing'):
        return [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])]  # [x, y]
    return [1.0, 1.0]

def get_slice_thickness(ds, next_ds=None):
    """Estimate slice thickness (Z spacing)."""
    if hasattr(ds, 'SliceThickness'):
        return float(ds.SliceThickness)
    if next_ds:
        return np.linalg.norm(np.array(ds.ImagePositionPatient) - np.array(next_ds.ImagePositionPatient))
    return 1.0

def construct_affine(first_ds, last_ds, Z_slices, pixel_spacing, slice_thickness):
    """Construct affine matrix based on orientation and spacing."""
    x_dir = np.array(first_ds.ImageOrientationPatient[3:])  # Column direction
    y_dir = np.array(first_ds.ImageOrientationPatient[:3])   # Row direction

    x_dir = -x_dir  # Reverse x direction vector
    y_dir = -y_dir  # Reverse y direction vector

    dx = pixel_spacing[0]  # Pixel spacing along x-axis
    dy = pixel_spacing[1]  # Pixel spacing along y-axis

    first_pos = np.array(first_ds.ImagePositionPatient)
    last_pos = np.array(last_ds.ImagePositionPatient)
    z_dir = (last_pos - first_pos) / (Z_slices - 1)

    R = np.eye(3)
    R[:, 0] = x_dir * dx
    R[:, 1] = y_dir * dy
    R[:, 2] = z_dir

    affine = from_matvec(R, first_pos)
    return affine

def dcm_to_nii(dicom_folder, nii_file_path):
    """Convert a folder of DICOM slices to NIfTI format."""
    dicom_files = get_sorted_slices(dicom_folder)
    if not dicom_files:
        raise ValueError("No valid DICOM files found.")

    first_ds = dcmread(dicom_files[0])
    last_ds = dcmread(dicom_files[-1])

    shape_ref = dcmread(dicom_files[0]).pixel_array.shape
    for f in dicom_files:
        ds = dcmread(f)
        assert ds.pixel_array.shape == shape_ref, f"Slice size mismatch: {f}"

    slices = []
    for f in dicom_files:
        ds = dcmread(f)
        slices.append(ds.pixel_array.T)  # Transpose each slice to correct (X,Y) order

    # Build volume
    volume = np.stack(slices, axis=0).astype(np.int16)  # shape: (S, Y, X)
    volume = np.transpose(volume, (1, 2, 0))  # shape: (Y, X, S)
    volume = np.transpose(volume, (1, 0, 2))  # shape: (X, Y, S)

    pixel_spacing = get_pixel_spacing(first_ds)
    slice_thickness = get_slice_thickness(first_ds, dcmread(dicom_files[1]) if len(dicom_files) > 1 else None)

    affine = construct_affine(first_ds, last_ds, volume.shape[2], pixel_spacing, slice_thickness)

    # Skip reorientation step; directly save original affine and volume
    nii_img = nib.Nifti1Image(volume, affine)

    nib.save(nii_img, nii_file_path)
    print(f"Conversion completed, file saved to: {nii_file_path}")

if __name__ == "__main__":
    dicom_folder = "./dcm/20240201000938/201"//Replace with the path of your dicom folder
    nii_file_path = "./nii/volume.nii"
    os.makedirs(os.path.dirname(nii_file_path), exist_ok=True)
    dcm_to_nii(dicom_folder, nii_file_path)
