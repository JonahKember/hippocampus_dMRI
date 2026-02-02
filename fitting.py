import os
import numpy as np
import nibabel as nib

from dipy.reconst import dki
from dipy.data import get_sphere


def fit_DKT_params(config):
    '''Crop and upsample the diffusion-weighted imagesm then fit DKT and DTI and collect parameters.'''

    subject    = config['subject']
    dwi_path   = config['dwi']
    bvecs      = config['bvecs']
    bvals      = config['bvals']
    B0_to_T1   = config['B0_to_T1']

    for hemi in ['L','R']:

        # Define paths.
        crop_B0_path     = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-preproc_T1w.nii.gz'
        crop_volume_B0   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-subfields.nii.gz'
        crop_dwi_path    = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-upsampled_dwi.nii.gz'

        dt_path          = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_tensor.nii.gz'
        dkt_path         = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_tensor.nii.gz'

        adc_path         = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_ADC.nii.gz'
        fa_path          = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_FA.nii.gz'
        ad_path          = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_AD.nii.gz'
        rd_path          = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_RD.nii.gz'
        eigenval_path    = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvals.nii.gz'
        eigenvec_path    = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvecs.nii.gz'

        dkt_params_path = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz'


        # Transform cropped T1 to B0 space.
        crop_T1 = config[f'hipp_crop_T1_{hemi}']
        crop_volume = config[f'hipp_volume_{hemi}']

        os.system(f'mrtransform {crop_T1} {crop_B0_path} -linear {B0_to_T1} -force')
        os.system(f'mrtransform {crop_volume} {crop_volume_B0} -linear {B0_to_T1} -force')

        # Crop and upsample the diffusion weighted images to 0.5mm resolution.
        os.system(f'mrgrid {dwi_path} regrid -voxel 0.5 -template {crop_B0_path} {crop_dwi_path} -force')

        # Fit diffusion-tensor and diffusion-kurtosis-tensor on cropped/upsampled DWI image.
        os.system(f'dwi2tensor {crop_dwi_path} {dt_path} -dkt {dkt_path} -fslgrad {bvecs} {bvals} -force')

        # Get diffusion-tensor metrics.
        os.system(f'tensor2metric \
            -fa    {fa_path} \
            -adc   {adc_path} \
            -ad    {ad_path} \
            -rd    {rd_path} \
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


def fit_directional_kurtosis(config, sphere_name='symmetric362'):

    subject = config['subject']

    for hemi in ['L','R']:

        params_nii = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz')
        params = params_nii.get_fdata()

        # Get directions.
        sphere = get_sphere(name=sphere_name)
        n_dirs = len(sphere.theta)

        # Flatten voxels, calculate directional kurtosis, reshape.
        X, Y, Z, P = params.shape

        params_flat   = params.reshape(-1, P)
        kurtosis_flat = dki.apparent_kurtosis_coef(params_flat, sphere)
        kurtosis_vox  = kurtosis_flat.reshape(X, Y, Z, n_dirs)

        # Write directional kurtosis to numpy and NIFTI.
        kurtosis_path_npy = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-kurtosis_sphere-{sphere_name}.npy'
        kurtosis_path_nii = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-kurtosis_sphere-{sphere_name}.nii.gz'

        np.save(kurtosis_path_npy, kurtosis_vox)

        kurtosis_nii = nib.Nifti1Image(
            kurtosis_vox,
            affine=params_nii.affine,
            header=params_nii.header
        )
        nib.save(kurtosis_nii, kurtosis_path_nii)


def fit_DKT_metrics(config):

    subject = config['subject']

    for hemi in ['L','R']:

        params_nii = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz')
        params = params_nii.get_fdata()

        for metric, model_func in zip(
            ['MK','AK','RK','KFA'],
            [dki.mean_kurtosis, dki.axial_kurtosis, dki.radial_kurtosis, dki.kurtosis_fractional_anisotropy]
            ):

            # Flatten voxels, calculate directional kurtosis, reshape.
            X, Y, Z, P = params.shape

            params_flat   = params.reshape(-1, P)
            kurtosis_flat = model_func(params_flat)
            kurtosis_metric  = kurtosis_flat.reshape(X, Y, Z)

            # Write directional kurtosis to numpy and NIFTI.
            kurtosis_path_npy = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_{metric}.npy'
            kurtosis_path_nii = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_{metric}.nii.gz'

            np.save(kurtosis_path_npy, kurtosis_metric)

            kurtosis_nii = nib.Nifti1Image(
                kurtosis_metric,
                affine=params_nii.affine,
                header=params_nii.header
            )
            nib.save(kurtosis_nii, kurtosis_path_nii)



def fit_directional_diffusion(config, sphere_name='symmetric362'):

    subject = config['subject']

    # Get directions of sphere.
    sphere = get_sphere(name=sphere_name)
    theta = sphere.theta
    phi = sphere.phi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere_dirs = np.stack([x, y, z], axis=1)
    n_dirs = sphere_dirs.shape[0]

    for hemi in ['L','R']:

        # Get diffusion tensor eigenvectors/eigenvalues.
        eigenvals = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvals.nii.gz').get_fdata()
        eigenvecs = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DT_eigenvecs.nii.gz').get_fdata()
        n_vox = eigenvals.shape[0]

        vox_diffusion = np.zeros([n_vox, n_vox, n_vox, n_dirs])

        for x in range(n_vox):
            for y in range(n_vox):
                for z in range(n_vox):

                    vox_eigenvecs = eigenvecs[x,y,z,:].reshape(3,3)
                    vox_eigenvals = eigenvals[x,y,z,:]

                    # DIPY order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
                    # https://docs.dipy.org/dev/examples_built/reconstruction/reconst_dti.html

                    D = vox_eigenvecs @ np.diag(vox_eigenvals) @ vox_eigenvecs.T
                    dt_params = np.array([
                        D[0,0],
                        D[0,1],
                        D[1,1],
                        D[0,2],
                        D[1,2],
                        D[2,2]
                    ])

                    vox_diffusion[x,y,z,:] = dki.directional_diffusion(dt_params, sphere_dirs)


        vox_diffusion_path  = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-diffusion_sphere-{sphere_name}.nii.gz'
        params_nii = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz')

        nii = nib.Nifti1Image(
            vox_diffusion,
            affine=params_nii.affine,
            header=params_nii.header
        )
        nib.save(nii, vox_diffusion_path)