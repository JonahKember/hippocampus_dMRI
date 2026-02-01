import os
import numpy as np
import pandas as pd
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

