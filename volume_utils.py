import numpy as np
import nibabel as nib


def get_mask_info(mask_path):

    nii = nib.load(mask_path)
    data = nii.get_fdata()

    indices = np.column_stack(np.where(data > 0))
    coords  = nib.affines.apply_affine(nii.affine, indices)
    flat    = data[data > 0]

    return data, flat, indices, coords

def get_mask_data(mask_path):
    return get_mask_info(mask_path)[0]

def get_mask_data_flat(mask_path):
    return get_mask_info(mask_path)[1]

def get_mask_voxel_indices(mask_path):
    return get_mask_info(mask_path)[2]

def get_mask_voxel_coords(mask_path):
    return get_mask_info(mask_path)[3]