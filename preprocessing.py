import os
import numpy as np
import pandas as pd
import nibabel as nib

from dipy.data import get_sphere
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

            if hemi == 'L': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusLeft'})
            if hemi == 'R': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusRight'})

            surf_gii = nib.gifti.GiftiImage(darrays=[surf_faces_arr, surf_coords_arr], meta=meta)

            nib.save(surf_gii, f'output/sub-{subject}_hemi-{hemi}_space-B0_den-0p5mm_label-hipp_{surf_type}.surf.gii')



def _vertex_normals(vertices, faces):

    # Get vectors.
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Get normals of face (cross product).
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.linalg.norm(fn, axis=1, keepdims=True)

    # Vertex normals: sum face normals of adjacent faces.
    N = vertices.shape[0]
    vn = np.zeros((N, 3), dtype=np.float64)

    for i in range(3):
        np.add.at(vn, faces[:, i], fn)

    normals = vn / np.linalg.norm(vn, axis=1, keepdims=True)

    return normals


def _cartesian_to_spherical(vectors):

    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    phi   = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1, 1))

    return theta, phi


def create_surface_normals(config):

    subject = config['subject']

    for surface_type in ['midthickness','inner','outer']:
        for hemi in ['L','R']:

            surf_path    = f'output/sub-{subject}_hemi-{hemi}_space-B0_den-0p5mm_label-hipp_{surface_type}.surf.gii'
            normals_path = f'output/sub-{subject}_hemi-{hemi}_space-B0_label-{surface_type}_desc-surface_normals.csv'

            surf = nib.load(surf_path)
            faces    = surf.darrays[0].data
            vertices = surf.darrays[1].data

            normals    = _vertex_normals(vertices, faces)
            theta, phi = _cartesian_to_spherical(normals)

            normals_df = pd.DataFrame({
                'x':normals[:,0],
                'y':normals[:,1],
                'z':normals[:,2],
                'theta':theta,
                'phi':phi
            })

            normals_df.to_csv(normals_path, index=False)



def create_surface_to_sphere_angles(config, sphere_name='symmetric362'):

    subject = config['subject']

    # Get directions of sphere.
    sphere = get_sphere(name=sphere_name)
    theta = sphere.theta
    phi = sphere.phi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere_dirs = np.stack([x, y, z], axis=1)


    for hemi in ['L','R']:

        # Get directions of surface-normals.
        surf_normals_df = pd.read_csv(f'output/sub-{subject}_hemi-{hemi}_space-B0_label-inner_desc-surface_normals.csv')
        surf_normals = surf_normals_df[['x','y','z']].to_numpy()

        # Calculate difference in angle between sphere-directions and surface-normals.
        angle_diffs = np.zeros([len(sphere_dirs), len(surf_normals)])

        for sphere_idx, sphere_dir in enumerate(sphere_dirs):
            for normal_idx, normal_dir in enumerate(surf_normals):

                cos_theta = np.clip(np.dot(sphere_dir, normal_dir), -1, 1)
                angle_diffs[sphere_idx, normal_idx] = np.arccos(cos_theta) * 180/np.pi

        np.save(f'output/sub-{subject}_hemi-{hemi}_label-inner_desc-surface_normal_direction.npy', angle_diffs)


        # Group sphere-directions into: normal, tangential, or oblique based on angle to surface-normal.
        n_vertices = len(surf_normals)

        vertex_direction = []
        for vertex in range(n_vertices):

            dir_group = np.zeros(len(sphere_dirs), dtype='<U10')

            normal     = (angle_diffs[:,vertex] <= 30) | (angle_diffs[:,vertex] >= 150)
            tangential = (angle_diffs[:,vertex] >= 60) & (angle_diffs[:,vertex] <= 120)
            oblique    = (normal == False) & (tangential == False)

            dir_group[normal]     = 'normal'
            dir_group[tangential] = 'tangential'
            dir_group[oblique]    = 'oblique'

            vertex_direction.append(dir_group)

        vertex_direction = np.array(vertex_direction)

        np.save(f'output/sub-{subject}_hemi-{hemi}_label-inner_desc-surface_normal_group.npy', vertex_direction)
