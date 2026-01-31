import os
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine


def create_B0_surface(config):
    """Transform hippocampal midthickness from T1 to B0 space."""

    subject   = config['subject']
    B0_to_T1  = np.loadtxt(config['B0_to_T1'])

    faces_intent  = nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
    coords_intent = nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    for surf_type in ['midthickness','inner','outer']:
        for hemi in ['L','R']:

            hipp_surf = config[f'hipp_{surf_type}_{hemi}']
            hipp_surf_gii = nib.load(hipp_surf)

            # Match GIFTI darrays to faces/vertices.
            for darray in hipp_surf_gii.darrays:

                if darray.intent == coords_intent: 
                    surf_coords = darray.data
                
                elif darray.intent == faces_intent:
                    surf_faces = darray.data

            # Transform hippocampal midthickness from T1 to B0 space.
            T1_to_B0 = np.linalg.inv(B0_to_T1)
            surf_coords_B0 = apply_affine(T1_to_B0, surf_coords)
            surf_coords_B0 = surf_coords_B0.astype(np.float32)

            # Create GIFTI for hippocampal surface in B0 space.
            surf_faces_arr  = nib.gifti.GiftiDataArray(surf_faces, intent=faces_intent)
            surf_coords_arr = nib.gifti.GiftiDataArray(surf_coords_B0, intent=coords_intent)

            surf_gii = nib.gifti.GiftiImage(darrays=[surf_faces_arr, surf_coords_arr])

            nib.save(surf_gii, f'output/sub-{subject}_hemi-{hemi}_space-B0_den-0p5mm_label-hipp_{surf_type}.surf.gii')



def get_DKT_params(config):
    '''Crop and upsample the diffusion-weighted imagesm then fit DKT and DTI and collect parameters.'''

    subject    = config['subject']
    dwi_path   = config['dwi']
    bvecs      = config['bvecs']
    bvals      = config['bvals']
    B0_to_T1   = config['B0_to_T1']

    for hemi in ['L','R']:

        # Define paths.
        crop_B0         = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-preproc_T1w.nii.gz'
        crop_dwi_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-upsampled_dwi.nii.gz'

        dt_path         = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_tensor.nii.gz'
        dkt_path        = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_tensor.nii.gz'

        adc_path        = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_ADC.nii.gz'
        fa_path         = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_FA.nii.gz'
        eigenval_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvals.nii.gz'
        eigenvec_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvecs.nii.gz'

        dkt_params_path = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz'


        # Transform cropped T1 to B0 space.
        crop_T1 = config[f'hipp_crop_T1_{hemi}']
        os.system(f'mrtransform {crop_T1} {crop_B0} -linear {B0_to_T1} -force')

        # Crop and upsample the diffusion weighted images to 0.5mm resolution.
        os.system(f'mrgrid {dwi_path} regrid -voxel 0.5 -template {crop_B0} {crop_dwi_path} -force')

        # Fit diffusion-tensor and diffusion-kurtosis-tensor on cropped/upsampled DWI image.
        os.system(f'dwi2tensor {crop_dwi_path} {dt_path} -dkt {dkt_path} -fslgrad {bvecs} {bvals} -force')

        # Get diffusion-tensor metrics.
        os.system(f'tensor2metric \
            -fa    {fa_path} \
            -adc   {adc_path} \
            -value {eigenval_path} \
            -vector {eigenvec_path} -num 1,2,3 -modulate none \
            {dt_path} \
            -force'
        )

        # Merge all DKT params expected by DIPY.
        dt_eigenvals = nib.load(eigenval_path).get_fdata()
        dt_eigenvecs = nib.load(eigenvec_path).get_fdata()
        dkt_params = nib.load(dkt_path).get_fdata()

        params = np.concatenate([
            dt_eigenvals,
            dt_eigenvecs,
            dkt_params
            ], axis=-1
        )

        # Write DKT parameters to 4D NIFTI.
        template_nii = nib.load(crop_dwi_path)
        params_nii = nib.Nifti1Image(
            params,
            affine=template_nii.affine,
            header=template_nii.header
        )

        nib.save(params_nii, dkt_params_path)
