import os
import numpy as np
import nibabel as nib
import io_utils, surface_utils, volume_utils

from nilearn import image
from scipy.spatial import cKDTree
from nibabel.affines import apply_affine
from dipy.segment.tissue import TissueClassifierHMRF



def transform_surfaces(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        for surface_path in ['midthickness','inner','outer']:

            B0_to_T1 = paths['B0_to_T1']
            surf_T1  = paths[f'{surface_path}']
            surf_B0  = paths[f'{surface_path}_space-B0']

            vertex_faces  = surface_utils.get_vertex_faces(surf_T1)
            vertex_coords = surface_utils.get_vertex_coords(surf_T1)

            B0_to_T1_array = np.loadtxt(B0_to_T1)
            T1_to_B0       = np.linalg.inv(B0_to_T1_array)
            vertex_coords_B0 = apply_affine(T1_to_B0, vertex_coords)
            vertex_coords_B0 = vertex_coords_B0.astype(np.float32)

            gii = surface_utils.create_surface_gii(vertex_faces, vertex_coords_B0, hemi)
            nib.save(gii, surf_B0)

    return


def transform_volumes(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)
        B0_to_T1 = paths['B0_to_T1']

        for volume_path in ['T1','subfields']:

            volume_T1 = paths[volume_path]
            volume_B0 = paths[f'{volume_path}_space-B0']

            os.system(f'mrtransform {volume_T1} {volume_B0} -linear {B0_to_T1} -force')

    return


def upsample_DWI(config):
    '''Crop and upsample the diffusion weighted images to 0.5mm resolution.'''

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        input    = paths['dwi']
        output   = paths['dwi_space-B0']
        template = paths['T1_space-B0']

        os.system(f'mrgrid {input} regrid -voxel 0.5 -template {template} {output} -force')

    return


def create_mask(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        subfields = paths['subfields_space-B0']
        dwi       = paths['dwi_space-B0']
        mask      = paths['mask']

        os.system(f'mrgrid {subfields} regrid -template {dwi} -interp nearest {mask} -force')

        mask_nii  = nib.load(mask)
        mask_data = mask_nii.get_fdata()
        mask_data[mask_data > 0] = 1

        mask_nii = nib.Nifti2Image(mask_data, mask_nii.affine, mask_nii.header)
        nib.save(mask_nii, mask)

    return


def create_distance_volume(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        outer = paths['outer_space-B0']
        vertex_mm = surface_utils.get_vertex_coords(outer)

        mask_nii      = nib.load(paths['mask'])
        mask_data     = volume_utils.get_mask_data(paths['mask'])
        voxel_indices = volume_utils.get_mask_voxel_indices(paths['mask'])
        voxel_coords  = volume_utils.get_mask_voxel_coords(paths['mask'])

        vertex_tree  = cKDTree(vertex_mm)
        distances, _ = vertex_tree.query(voxel_coords)

        distance_vol = np.zeros_like(mask_data)
        for dist, vox_idx in zip(distances, voxel_indices):
            x, y, z = vox_idx
            distance_vol[x,y,z] = dist

        distance_vol[distance_vol == 0] = np.nan

        distance_path = paths['distance']
        distance_nii = nib.Nifti2Image(distance_vol, mask_nii.affine, mask_nii.header)
        nib.save(distance_nii, distance_path)

    return



def create_tissue_segmentation(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        img_path    = paths['T1_space-B0']
        tissue_path = paths['tissue_seg']
        pvol_path   = paths['pvol']
        ref_path    = paths['mask']


        ref_nii  = nib.load(ref_path)
        img_nii  = nib.load(img_path)
        img_data = img_nii.get_fdata()

        # Segment T1 (gray/white/CSF).
        tissue_classifier = TissueClassifierHMRF()
        _, final_seg, partial_vol = tissue_classifier.classify(img_data, nclasses=3, beta=.1)

        # Resample to 0.5mm resolution, write tissue segmentation to NIFTI.
        tissue_nii = nib.Nifti2Image(
            final_seg,
            affine=img_nii.affine,
            header=img_nii.header
        )

        nii_resampled  = image.resample_to_img(tissue_nii, ref_path)
        data_resampled = np.int32(nii_resampled.get_fdata())

        nii = nib.Nifti2Image(
            data_resampled,
            affine=ref_nii.affine,
            header=ref_nii.header
        )
        nib.save(nii, tissue_path)

        # Resample to 0.5mm resolution, write partial volume to NIFTI.
        pvol_nii = nib.Nifti2Image(
            partial_vol,
            affine=img_nii.affine,
            header=img_nii.header
        )
        pvol_nii_resampled = image.resample_to_img(pvol_nii, ref_path)
        nib.save(pvol_nii_resampled, pvol_path)

