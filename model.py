import pickle
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.spatial import cKDTree
from dipy.data import get_sphere
from dipy.reconst import dki, dki_micro



def fit_model(config):

    subject = config['subject']

    for hemi in ['L','R']:
        _get_voxel_directional_diffusion(subject, hemi)
        _get_vertex_sphere_angles(subject, hemi)
        _get_vertex_directional_diffusion(subject, hemi)



def _get_voxel_directional_diffusion(subject, hemi, sphere_name='symmetric362'):
    '''
    For each voxel in the hippocampus, assign a sphere direwcion

    '''

    # Get directions of sphere.
    sphere = get_sphere(name=sphere_name)
    theta = sphere.theta
    phi = sphere.phi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    sphere_dirs  = np.stack([x, y, z], axis=1)
    n_directions = len(sphere_dirs)

    # Get indices of masked voxels.
    _, voxel_indices, _ = _get_voxel_mask_info(subject, hemi)

    # For voxels within mask, estimate compartment- and direction- specific diffusion.
    for hemi in ['L','R']:

        params_nii = nib.load(f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-DKT_params.nii.gz')
        params = params_nii.get_fdata()

        awf        = np.zeros(len(voxel_indices))                  # Axonal water fracion
        extra_diff = np.zeros([len(voxel_indices), n_directions])  # Extra-cellular diffusion.
        intra_diff = np.zeros([len(voxel_indices), n_directions])  # Intra-cellular diffusion.

        for n, voxel_idx in enumerate(voxel_indices):

            x, y, z = voxel_idx
            voxel_params = params[x,y,z,:]

            extra_tensor, intra_tensor = dki_micro.diffusion_components(voxel_params)

            awf[n] = dki_micro.axonal_water_fraction(voxel_params)
            extra_diff[n,:] = dki.directional_diffusion(extra_tensor, sphere_dirs)
            intra_diff[n,:] = dki.directional_diffusion(intra_tensor, sphere_dirs)

        np.save(f'output_model/sub-{subject}_hemi-{hemi}_sphere-{sphere_name}_desc-intra_diffusion.npy', intra_diff)
        np.save(f'output_model/sub-{subject}_hemi-{hemi}_sphere-{sphere_name}_desc-extra_diffusion.npy', extra_diff)
        np.save(f'output_model/sub-{subject}_hemi-{hemi}_sphere-{sphere_name}_desc-awf.npy', awf)




def _get_vertex_sphere_angles(subject, hemi, sphere_name='symmetric362', surf_type='inner'):
    '''
    For each voxel in the hippocampus, assign a sphere direwcion

    '''

    # Get directions of sphere.
    sphere = get_sphere(name=sphere_name)
    theta = sphere.theta
    phi = sphere.phi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere_dirs  = np.stack([x, y, z], axis=1)

    # Get directions of surface-normals.
    surf_normals_df = pd.read_csv(f'output/sub-{subject}_hemi-{hemi}_space-B0_label-{surf_type}_desc-surface_normals.csv')
    surf_normals = surf_normals_df[['x','y','z']].to_numpy()

    # Calculate difference in angle between sphere-directions and surface-normals.
    angle_diffs = np.zeros([len(sphere_dirs), len(surf_normals)])

    for sphere_idx, sphere_dir in enumerate(sphere_dirs):
        for normal_idx, normal_dir in enumerate(surf_normals):

            cos_theta = np.clip(np.dot(sphere_dir, normal_dir), -1, 1)
            angle_diffs[sphere_idx, normal_idx] = np.arccos(cos_theta) * 180/np.pi


    # Group surface-sphere angle differences as either normal or tangential.
    vertex_angle_labels = []
    for vertex in range(len(surf_normals)):

        angle_label = np.zeros(len(sphere_dirs), dtype='<U10')

        normal     = (angle_diffs[:,vertex] <= 45) | (angle_diffs[:,vertex] >= 135)
        tangential = (angle_diffs[:,vertex] >= 45) & (angle_diffs[:,vertex] <= 135)

        angle_label[normal] = 'normal'
        angle_label[tangential] = 'tangential'

        vertex_angle_labels.append(angle_label)

    vertex_angle_labels = np.array(vertex_angle_labels)
    np.save(f'output_model/sub-{subject}_hemi-{hemi}_sphere-{sphere_name}_surf-{surf_type}_desc-surface_sphere_angle_labels.npy', vertex_angle_labels)




def _get_vertex_directional_diffusion(subject, hemi, sigma_mm=2, radius_mm=6, sphere_name='symmetric362', surf_type='inner'):


    surf_path            = f'output/sub-{subject}_hemi-{hemi}_space-B0_den-0p5mm_label-hipp_{surf_type}.surf.gii'
    vertex_sphere_angles = np.load(f'output_model/sub-{subject}_hemi-{hemi}_sphere-{sphere_name}_surf-{surf_type}_desc-surface_sphere_angle_labels.npy')
    vox_extra_diff       = np.load(f'output_model/sub-{subject}_hemi-{hemi}_sphere-symmetric362_desc-extra_diffusion.npy')
    vox_intra_diff       = np.load(f'output_model/sub-{subject}_hemi-{hemi}_sphere-symmetric362_desc-intra_diffusion.npy')
    awf                  = np.load(f'output_model/sub-{subject}_hemi-{hemi}_sphere-symmetric362_desc-awf.npy')


    # Get surface info.
    surf       = nib.load(surf_path)
    vertex_mm  = surf.darrays[1].data
    n_vertices   = vertex_sphere_angles.shape[0]

    # Get coordinates of masked voxels.
    mask_data_flat, _, voxel_mm = _get_voxel_mask_info(subject, hemi)
    voxel_tree = cKDTree(voxel_mm)

    vertex_values = {
        'awf':np.full(n_vertices, np.nan),
        'intra_normal':np.full(n_vertices, np.nan),
        'intra_tangential':np.full(n_vertices, np.nan),
        'extra_normal':np.full(n_vertices, np.nan),
        'extra_tangential':np.full(n_vertices, np.nan)
    }

    for idx, mm in enumerate(vertex_mm):

        # Find nearby voxels (within radius).
        nearby_vox = voxel_tree.query_ball_point(mm, r=radius_mm)
        nearby_vox_coords = voxel_mm[nearby_vox]
        nearby_vox_distances = np.linalg.norm(nearby_vox_coords - mm, axis=1)

        # Get voxel weights based on partial-volume and gaussian-weighted distance.
        distance_W = np.exp(-0.5 * (nearby_vox_distances / sigma_mm)**2)
        partial_volume_W = mask_data_flat[nearby_vox]
        W = distance_W * partial_volume_W

        vertex_values[f'awf'][idx] = np.average(awf[nearby_vox], weights=W)

        # For all nearby voxels, calculate the mean diffusion in the specified directions.
        for label in ['normal','tangential']:

            # Get directions on sphere aligned with 'label'.
            include_dirs = np.argwhere(vertex_sphere_angles[idx,:] == label).flatten()

            dir_intra_diff = vox_intra_diff[nearby_vox,:][:,include_dirs].mean(axis=1)
            dir_extra_diff = vox_extra_diff[nearby_vox,:][:,include_dirs].mean(axis=1)

            vertex_values[f'intra_{label}'][idx] = np.average(dir_intra_diff, weights=W)
            vertex_values[f'extra_{label}'][idx] = np.average(dir_extra_diff, weights=W)


    # Write vertex-wise model outputs to binary.
    with open(f'output_model/sub-{subject}_hemi-{hemi}_surf-{surf_type}_desc-model_params.{hemi}.pkl','wb') as file:
        pickle.dump(vertex_values, file)


    # Write vertex-wise model outputs to GIFTI.
    intent = nib.nifti1.intent_codes['NIFTI_INTENT_NONE']
    if hemi == 'L': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusLeft'})
    if hemi == 'R': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusRight'})

    darrays = []
    for map_name, data in vertex_values.items():
        darray = nib.gifti.GiftiDataArray(np.array(data, dtype='float32'), intent=intent)
        darray.meta = nib.gifti.GiftiMetaData({'Name':map_name})
        darrays.append(darray)

    gii = nib.GiftiImage(darrays=darrays, meta=meta)
    nib.save(gii, f'output_model/sub-{subject}_hemi-{hemi}_surf-{surf_type}_desc-model_params.{hemi}.func.gii')




def _get_voxel_mask_info(subject, hemi):
    '''Get coordinates of masked voxels.

    Returns
    -------
    mask_data_flat: 1D boolean array of masked voxels.
    voxel_indices: indices of voxels in image data (x, y, z).
    voxel_mm: xyz coordinates in image space (mm).
    '''

    pvol_path = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-partial_volume.nii.gz'

    mask_nii = nib.load(pvol_path)
    mask_data = mask_nii.get_fdata()
    mask_affine = mask_nii.affine
    mask_data_flat = mask_data[mask_data > 0]

    voxel_indices = np.column_stack(np.where(mask_data > 0))
    voxel_mm = nib.affines.apply_affine(mask_affine, voxel_indices)

    return mask_data_flat, voxel_indices, voxel_mm

