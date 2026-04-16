import numpy as np
import nibabel as nib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import SpectralEmbedding
from hippocampus_dMRI_T1_space import io_utils


def _get_log_ratio(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        nii_norm = nib.load(paths['diffusion_normal'])
        nii_tan  = nib.load(paths['diffusion_tangential'])

        diffusion_norm = nii_norm.get_fdata()
        diffusion_tan  = nii_tan.get_fdata()

        diffusion_norm[diffusion_norm == 0] = 1
        diffusion_tan[diffusion_tan == 0] = 1

        log_ratio = np.log(diffusion_tan / diffusion_norm)

        log_ratio_nii = nib.Nifti1Image(log_ratio, affine=nii_norm.affine)
        nib.save(log_ratio_nii, paths['diffusion_log_ratio'])


def get_spatial_gradient(config):

    _get_log_ratio(config)

    for hemi in ['L','R']:

        # Load data.
        paths = io_utils.get_paths(config, hemi)

        mask_nii = nib.load(paths['mask_refined'])
        mask = np.round(mask_nii.get_fdata())

        log_ratio = nib.load(paths['diffusion_log_ratio']).get_fdata()
        kurtosis  = nib.load(paths['DKT_MK']).get_fdata()

        log_ratio_flat = log_ratio[mask > 0]
        kurtosis_flat  = kurtosis[mask > 0]

        # Prepare data.
        X = np.array([log_ratio_flat, kurtosis_flat]).T
        X = np.nan_to_num(X, 0)
        X = StandardScaler().fit_transform(X)

        # Gradient modeling.
        gradient_mdl = SpectralEmbedding(
            n_components=1,
            affinity='nearest_neighbors',
            n_neighbors=30
        )

        gradient_score = gradient_mdl.fit_transform(X)
        gradient_score = MinMaxScaler().fit_transform(gradient_score).flatten()

        gradient_vol = np.zeros(mask.shape)
        gradient_vol[mask > 0] = gradient_score
        gradient_nii = nib.Nifti1Image(gradient_vol, affine=mask_nii.affine)
        nib.save(gradient_nii, paths['spatial_gradient'])

