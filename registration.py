import os
import subprocess
import numpy as np
import nibabel as nib

from hippocampus_dMRI_T1_space import io_utils
from pathlib import Path
from nilearn.image import resample_img
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.noise_estimate import estimate_sigma


def refine_mask(config):
    '''Perform a more focused transformation of anatomical hippocampal mask to diffusion-weighted data.'''

    for hemi in ['L','R']:
        paths = io_utils.get_paths(config, hemi)

        _create_mean_B0(paths['dwi_upsampled'], paths['bvals'], paths['mean_B0'])
        _adapative_smoothing(paths['mean_B0'], paths['mean_B0_smooth'])
        _create_rigid_transform(paths['subject'], hemi, paths['T1'], paths['mean_B0_smooth'])

        for prefix in ['T1','layers','mask','distance']:
            _apply_rigid_transform(
                paths['subject'],
                hemi,
                input=paths[prefix],
                output=paths[f'{prefix}_refined'],
                reference=paths['mean_B0_smooth'],
                type='vol'
            )

        for prefix in ['midthickness','inner','outer']:
            _apply_rigid_transform(
                paths['subject'],
                hemi,
                input=paths[prefix],
                output=paths[f'{prefix}_refined'],
                reference=paths['mean_B0_smooth'],
                type='surf'
            )

    return


def _create_rigid_transform(subject, hemi, anat_path, param_path, ants_dir='/opt/minc/1.9.18/bin'):

    anat_path = Path(anat_path)

    # Add ANTS to environment. 
    env = os.environ.copy()
    env['ANTSPATH'] = ants_dir
    env['PATH'] = f"{ants_dir}:{env.get('PATH','')}"

    cmd_reg = [
        'antsRegistrationSyNQuick.sh',
        '-d', '3',
        '-m', str(param_path),
        '-f', str(anat_path),
        '-t', 'r',
        '-o', f'{anat_path.parent}/sub-{subject}_hemi-{hemi}_refined_transform_',
        '-n', '5'
    ]
    subprocess.run(cmd_reg, check=True, env=env)

    return


def _apply_rigid_transform(subject, hemi, input, output, reference, type, ants_dir='/opt/minc/1.9.18/bin'):

    # ANTs environment
    env = os.environ.copy()
    env['ANTSPATH'] = ants_dir
    env['PATH'] = f"{ants_dir}:{env.get('PATH','')}"

    # Get transform files.
    input_path = Path(input)
    affine  = f'{input_path.parent}/sub-{subject}_hemi-{hemi}_refined_transform_0GenericAffine.mat'

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input),
        '-r', str(reference),
        '-o', str(output),
        '-t', f'[{affine},1]'
    ]

    if type == 'vol':
        subprocess.run(cmd, check=True, env=env)

    if type == 'surf':
        affine = affine.replace('/surf/','/anat/')
        affine_txt = affine.replace('/anat/','/surf/').replace('transform_0GenericAffine.mat','surface_transform.txt')
        os.system(f'ConvertTransformFile 3 {affine} {affine_txt} --homogeneousMatrix')
        os.system(f'wb_command -surface-apply-affine {input} {affine_txt} {output}')

    return



def _create_mean_B0(dwi_path, bvals_path, mean_b0_path, B0_threshold=25):

    dwi_nii = nib.load(dwi_path)
    bvals   = np.loadtxt(bvals_path)
    
    B0_idx = np.argwhere(np.abs(bvals) < B0_threshold).flatten()

    dwi_data = dwi_nii.get_fdata()
    mean_B0_data = dwi_data[:,:,:,B0_idx].mean(axis=-1)
    mean_B0_nii  = nib.Nifti1Image(mean_B0_data, affine=dwi_nii.affine)
    nib.save(mean_B0_nii, mean_b0_path)



def _adapative_smoothing(nii_path, output_path, patch_radii=[3,3], block_radii=[5,10]):

    nii   = nib.load(nii_path)
    data  = nii.get_fdata()
    sigma = estimate_sigma(data, N=4)

    patch_small, patch_large = patch_radii
    block_small, block_large = block_radii

    den_small = nlmeans(
        data, sigma=sigma,
        patch_radius=patch_small,
        block_radius=block_small,
        rician=True
    )

    den_large = nlmeans(
        data, sigma=sigma,
        patch_radius=patch_large,
        block_radius=block_large,
        rician=True
    )

    den_final = adaptive_soft_matching(data, den_small, den_large, sigma[0])
    nii_cleaned = nib.Nifti1Image(den_final, nii.affine) 
    nib.save(nii_cleaned, output_path)
