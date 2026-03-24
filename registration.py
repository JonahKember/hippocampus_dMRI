import io_utils
import numpy as np
import nibabel as nib

from nilearn.image import resample_img
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D


def refine_mask(config):
    '''Perform a more focused transformation of anatomical hippocampal mask to diffusion-weighted data.'''

    for hemi in ['L','R']:
        paths = io_utils.get_paths(config, hemi)

        _create_mean_B0(paths['dwi_space-B0'], paths['bvals'], paths['mean_B0'])
        mask_transformed_nii = _high_resolution_transform(
            anat_path=paths['T1_space-B0'],
            param_path=paths['mean_B0'],
            mask_path=paths['mask']
        )
        nib.save(mask_transformed_nii, paths['mask_refined'])

    return


def _high_resolution_transform(anat_path, param_path, mask_path):

    anat  = nib.load(anat_path)
    param = nib.load(param_path)
    mask  = nib.load(mask_path)

    anat_data = anat.get_fdata().astype(np.float32)

    # Regrid parameter volume and mask to anatomical space.
    resampled_imgs = {}
    for img, interp, name in [(param,'linear','param'), (mask,'nearest','mask')]:

        resampled_imgs[name] = resample_img(
            img,
            target_affine=anat.affine,
            target_shape=anat_data.shape,
            interpolation=interp
        )

    param_data = resampled_imgs['param'].get_fdata().astype(np.float32)
    mask_data  = (resampled_imgs['mask'].get_fdata() > 0).astype(np.uint8)

    # Create rigid transform from parameter space to anat.
    metric = MutualInformationMetric()
    affreg = AffineRegistration(
        metric=metric,
        level_iters=[500, 200, 100],
        sigmas=[1.0,0.5,0.0],
        factors=[2,1,1]
    )

    opt_map = affreg.optimize(
        static=anat_data,
        moving=param_data,
        transform=RigidTransform3D(),
        params0=None,
        static_grid2world=anat.affine,
        moving_grid2world=anat.affine
    )

    # Apply inverse-transform to mask.
    mask_transformed = opt_map.transform_inverse(mask_data)
    mask_transformed_nii = nib.Nifti1Image(mask_transformed, anat.affine)

    # Resample refined mask to original grid.
    mask_transformed_nii = resample_img(
        mask_transformed_nii,
        target_affine=mask.affine,
        target_shape=mask.shape,
        interpolation='nearest'
    )

    return mask_transformed_nii


def _create_mean_B0(dwi_path, bvals_path, mean_b0_path, B0_threshold=25):

    dwi_nii = nib.load(dwi_path)
    bvals   = np.loadtxt(bvals_path)
    
    B0_idx = np.argwhere(np.abs(bvals) < B0_threshold).flatten()

    dwi_data = dwi_nii.get_fdata()
    mean_B0_data = dwi_data[:,:,:,B0_idx].mean(axis=-1)
    mean_B0_nii  = nib.Nifti1Image(mean_B0_data, affine=dwi_nii.affine)
    nib.save(mean_B0_nii, mean_b0_path)
