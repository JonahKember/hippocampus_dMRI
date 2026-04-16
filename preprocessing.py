import os
import numpy as np
import nibabel as nib
from hippocampus_dMRI_T1_space import io_utils

from nilearn import image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, binary_dilation
from nibabel.affines import apply_affine
from dipy.segment.tissue import TissueClassifierHMRF


def upsample_DWI(config):
    '''Crop and upsample the diffusion weighted images to 0.5mm resolution.'''

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        input    = paths['dwi']
        output   = paths['dwi_upsampled']
        template = paths['T1']

        os.system(f'mrgrid {input} regrid -voxel 0.5 -template {template} {output} -force')

    return


def create_mask(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        subfields = paths['subfields']
        dwi       = paths['dwi_upsampled']
        mask      = paths['mask']

        os.system(f'mrgrid {subfields} regrid -template {dwi} -interp nearest {mask} -force')

        mask_nii  = nib.load(mask)
        mask_data = mask_nii.get_fdata()
        mask_data[mask_data > 0] = 1

        # Dilate mask to account for under-segmentation by HippUnfold in the anterior/superior direction.
        structure = np.zeros((3,3,3))
        structure[:, 1:, 1:] = 1
        dilated = binary_dilation(mask_data, structure=structure)

        mask_nii = nib.Nifti1Image(dilated.astype(np.uint8), mask_nii.affine, mask_nii.header)
        nib.save(mask_nii, mask)


    return


def create_layer_mask(config, hippunfold_srlm_idx=7):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        layers    = paths['layers']
        subfields = paths['subfields']

        # Load subfields mask
        subfields_nii = nib.load(subfields)
        affine = subfields_nii.affine
        data   = subfields_nii.get_fdata()

        # Create binary masks.
        mask = np.full(data.shape, np.nan, dtype=np.float32)
        hipp_mask = data > 0
        srlm_mask = data == hippunfold_srlm_idx

        dist_to_srlm = distance_transform_edt(~srlm_mask, sampling=np.abs(affine[:3, :3].diagonal()))
        dist_to_outside = distance_transform_edt(hipp_mask, sampling=np.abs(affine[:3, :3].diagonal()))

        mask[hipp_mask & (~srlm_mask) & (dist_to_outside < dist_to_srlm)] = 1
        mask[srlm_mask] = 2

        nib.save(nib.Nifti1Image(mask, affine), layers)



def create_distance_volume(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        mask_nii = nib.load(paths['mask'])
        mask_data = mask_nii.get_fdata()

        # Create tree of voxels outside the mask.
        mask_data_full = mask_data.copy()
        mask_data_full[mask_data == 0] = 1
        mask_data_full[mask_data != 0] = 0

        voxel_indices = np.column_stack(np.where(mask_data_full))
        voxel_coords = nib.affines.apply_affine(mask_nii.affine, voxel_indices)
        voxel_tree  = cKDTree(voxel_coords)

        # Query the nearest distance of voxels within mask to voxels outside mask.
        mask_indices  = np.column_stack(np.where(mask_data))
        mask_coords  = nib.affines.apply_affine(mask_nii.affine, mask_indices)
        distances, _ = voxel_tree.query(mask_coords)

        # Write to volume.
        distance_vol = np.zeros_like(mask_data)
        for dist, vox_idx in zip(distances, mask_indices):
            x, y, z = vox_idx
            distance_vol[x,y,z] = dist

        distance_vol[distance_vol == 0] = np.nan
        distance_vol = distance_vol - np.nanmin(distance_vol)
        distance_nii = nib.Nifti1Image(distance_vol, mask_nii.affine, mask_nii.header)
        nib.save(distance_nii, paths['distance'])

    return


def create_tissue_segmentation(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        img_path    = paths['T1_refined']
        tissue_path = paths['tissue_seg']
        pvol_path   = paths['pvol']
        ref_path    = paths['mask_refined']


        ref_nii  = nib.load(ref_path)
        img_nii  = nib.load(img_path)
        img_data = img_nii.get_fdata()

        # Segment T1 (gray/white/CSF).
        tissue_classifier = TissueClassifierHMRF()
        _, final_seg, partial_vol = tissue_classifier.classify(img_data, nclasses=2, beta=.1)

        # Resample to 0.5mm resolution, write tissue segmentation to NIFTI.
        tissue_nii = nib.Nifti1Image(
            final_seg,
            affine=img_nii.affine,
            header=img_nii.header
        )

        nii_resampled  = image.resample_to_img(tissue_nii, ref_path)
        data_resampled = np.int32(nii_resampled.get_fdata())

        nii = nib.Nifti1Image(
            data_resampled,
            affine=ref_nii.affine,
            header=ref_nii.header
        )
        nib.save(nii, tissue_path)

        # Resample to 0.5mm resolution, write partial volume to NIFTI.
        pvol_nii = nib.Nifti1Image(
            partial_vol,
            affine=img_nii.affine,
            header=img_nii.header
        )
        pvol_nii_resampled = image.resample_to_img(pvol_nii, ref_path)
        nib.save(pvol_nii_resampled, pvol_path)
