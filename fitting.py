import os
import numpy as np
import nibabel as nib
import io_utils, volume_utils, surface_utils

from scipy.spatial import cKDTree
from dipy.data import get_sphere
from dipy.reconst import dki, dki_micro


def fit_diffusion_tensors(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        # Fit diffusion-tensor and diffusion-kurtosis-tensor on cropped/upsampled DWI image.
        dwi   = paths['dwi_space-B0']
        bvals = paths['bvals']
        bvecs = paths['bvecs']

        dt    = paths['diffusion_tensor']
        dkt   = paths['diffusion_kurtosis_tensor']

        os.system(f'dwi2tensor {dwi} {dt} -dkt {dkt} -fslgrad {bvecs} {bvals} -force')

        # Calculate diffusion-tensor metrics.
        os.system(f"tensor2metric \
            -fa    {paths['DT_fa']} \
            -adc   {paths['DT_adc']} \
            -ad    {paths['DT_ad']} \
            -rd    {paths['DT_rd']} \
            -value {paths['DT_eigenvals']} \
            -vector {paths['DT_eigenvecs']} -num 1,2,3 -modulate none \
            {paths['diffusion_tensor']} \
            -force"
        )

        # Merge all DKT params expected by DIPY.
        dt_vals    = nib.load(paths['DT_eigenvals']).get_fdata()
        dt_vecs    = nib.load(paths['DT_eigenvecs']).get_fdata()
        dkt_params = nib.load(paths['diffusion_kurtosis_tensor']).get_fdata()
        params     = np.concatenate([dt_vals, dt_vecs, dkt_params], axis=-1)

        # Write DKT parameters to 4D NIFTI.
        dwi_nii = nib.load(dwi)
        params_nii = nib.Nifti1Image(
            params,
            affine=dwi_nii.affine,
            header=dwi_nii.header
        )
        nib.save(params_nii, paths['DKT_params'])


        # Calculate DKT metrics.
        metric_dict = {
            'MK': dki.mean_kurtosis,
            'AK': dki.axial_kurtosis,
            'RK': dki.radial_kurtosis,
            'KFA': dki.kurtosis_fractional_anisotropy
        }

        for metric_name, metric_function in metric_dict.items():

            # Flatten voxels, calculate directional kurtosis, reshape.
            X, Y, Z, P = params.shape

            params_flat   = params.reshape(-1, P)
            kurtosis_flat = metric_function(params_flat)
            kurtosis_metric  = kurtosis_flat.reshape(X, Y, Z)

            # Write  kurtosis NIFTI.
            kurtosis_path = paths[f'DKT_{metric_name}']
            kurtosis_nii = nib.Nifti1Image(
                kurtosis_metric,
                affine=params_nii.affine,
                header=params_nii.header
            )
            nib.save(kurtosis_nii, kurtosis_path)

    return



def fit_neurite_volume_fraction(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        mask_path   = paths['mask']
        params_path = paths['DKT_params']
        nvf_path    = paths['neurite_volume_fraction']

        mask_nii      = nib.load(mask_path)
        voxel_indices = volume_utils.get_mask_voxel_indices(mask_path)
        params        = nib.load(params_path).get_fdata()

        awf = np.zeros_like(mask_nii.get_fdata())
        for voxel_idx in voxel_indices:

            x, y, z = voxel_idx
            voxel_params = params[x,y,z,:]
            awf[x, y, z] = dki_micro.axonal_water_fraction(voxel_params)

        # Write to NIFTI.
        awf_nii = nib.Nifti2Image(
            awf,
            affine=mask_nii.affine,
            header=mask_nii.header
        )
        nib.save(awf_nii, nvf_path)



def get_voxel_directional_diffusion(config):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        mask_path            = paths['mask']
        surf_path            = paths['midthickness_space-B0']
        eigenvals_path       = paths['DT_eigenvals']
        eigenvecs_path       = paths['DT_eigenvecs']

        diff_normal_path     = paths['diffusion_normal']
        diff_tangential_path = paths['diffusion_tangential']

        # Load eigenvalues/vectors of the diffusion tensor.
        eigenvals = nib.load(eigenvals_path).get_fdata()
        eigenvecs = nib.load(eigenvecs_path).get_fdata()

        # Initialize tree with vertex coordinates.
        vertex_faces   = surface_utils.get_vertex_faces(surf_path)
        vertex_coords  = surface_utils.get_vertex_coords(surf_path)
        vertex_normals = surface_utils.get_vertex_normals(vertex_faces, vertex_coords)
        vertex_tree = cKDTree(vertex_coords)

        # Get mask coordinates.
        mask_nii = nib.load(mask_path)
        mask_data, _, voxel_indices, voxel_coords = volume_utils.get_mask_info(mask_path)

        # Calculate voxel-wise diffusion normal and tangential to the hippocampal surface.
        diffusion_normal     = np.zeros_like(mask_data)
        diffusion_tangential = np.zeros_like(mask_data)
        for idx, vox_idx in enumerate(voxel_indices):

            # Get surface-normal of nearest vertex to voxel.
            _, nearby_vertex = vertex_tree.query(voxel_coords[idx])
            voxel_normal     = vertex_normals[nearby_vertex]

            # Get diffusion parallel and perpendicular to vertex normal.
            x, y, z = vox_idx
            eigval = eigenvals[x, y, z]
            eigvec = eigenvecs[x, y, z].reshape(3,3).T

            D_par  = np.sum(eigval * (eigvec.T @ voxel_normal)**2)
            D_perp = (np.sum(eigval) - D_par) / 2

            diffusion_normal[x, y, z]     = D_par
            diffusion_tangential[x, y, z] = D_perp

        # Write surface-normal diffusion to NIFTI.
        diffusion_normal_nii = nib.Nifti2Image(
            diffusion_normal,
            affine=mask_nii.affine,
            header=mask_nii.header
        )
        nib.save(diffusion_normal_nii, diff_normal_path)

        # Write surface-tangential diffusion to NIFTI.
        diffusion_tangential_nii = nib.Nifti2Image(
            diffusion_tangential,
            affine=mask_nii.affine,
            header=mask_nii.header
        )
        nib.save(diffusion_tangential_nii, diff_tangential_path)



def fit_sphere_kurtosis(config, sphere_name='symmetric362'):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        params_path          = paths['DKT_params']
        kurtosis_sphere_path = paths['kurtosis_sphere']


        params_nii = nib.load(params_path)
        params = params_nii.get_fdata()

        # Get directions.
        sphere = get_sphere(name=sphere_name)
        n_dirs = len(sphere.theta)

        # Flatten voxels, calculate directional kurtosis, reshape.
        X, Y, Z, P = params.shape

        params_flat   = params.reshape(-1, P)
        kurtosis_flat = dki.apparent_kurtosis_coef(params_flat, sphere)
        kurtosis_vox  = kurtosis_flat.reshape(X, Y, Z, n_dirs)

        # Write directional kurtosis to NIFTI.
        kurtosis_nii = nib.Nifti1Image(
            kurtosis_vox,
            affine=params_nii.affine,
            header=params_nii.header
        )
        nib.save(kurtosis_nii, kurtosis_sphere_path)



def fit_sphere_diffusion(config, sphere_name='symmetric362'):

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

        paths = io_utils.get_paths(config, hemi)

        mask_path        = paths['mask']
        eigenvals_path   = paths['DT_eigenvals']
        eigenvecs_path   = paths['DT_eigenvecs']
        diff_sphere_path = paths['diffusion_sphere']

        # Load eigenvalues/vectors of the diffusion tensor.
        eigenvals = nib.load(eigenvals_path).get_fdata()
        eigenvecs = nib.load(eigenvecs_path).get_fdata()

        # Get diffusion tensor eigenvectors/eigenvalues.
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

        mask = nib.load(mask_path)
        nii = nib.Nifti1Image(
            vox_diffusion,
            affine=mask.affine,
            header=mask.header
        )
        nib.save(nii, diff_sphere_path)