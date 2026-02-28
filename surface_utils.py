import numpy as np
import nibabel as nib

faces_intent  = nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
coords_intent = nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']


def get_vertex_faces(gii):

    if isinstance(gii, str):
        gii = nib.load(gii)

    faces_intent  = nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
    for darray in gii.darrays:
        if darray.intent == faces_intent: 
            vertex_faces = darray.data

    return vertex_faces


def get_vertex_coords(gii):

    if isinstance(gii, str):
        gii = nib.load(gii)

    for darray in gii.darrays:
        if darray.intent == coords_intent: 
            vertex_coords = darray.data

    return vertex_coords


def create_surface_gii(vertex_faces, vertex_coords, hemi):

    # Create GIFTI for hippocampal surface in B0 space.
    faces_darray  = nib.gifti.GiftiDataArray(vertex_faces, intent=faces_intent)
    coords_darray = nib.gifti.GiftiDataArray(vertex_coords, intent=coords_intent)

    if hemi == 'L': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusLeft'})
    if hemi == 'R': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusRight'})

    gii = nib.gifti.GiftiImage(darrays=[faces_darray, coords_darray], meta=meta)

    return gii


def get_vertex_normals(vertex_faces, vertex_coords):

    # Get face vectors.
    v0 = vertex_coords[vertex_faces[:, 0]]
    v1 = vertex_coords[vertex_faces[:, 1]]
    v2 = vertex_coords[vertex_faces[:, 2]]

    # Get normals of face vectors.
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Vertex normals: sum face normals of adjacent faces.
    n_vertex, n_dim = vertex_coords.shape
    vertex_normals = np.zeros((n_vertex, 3), dtype=np.float64)

    for i in range(n_dim):
        np.add.at(vertex_normals, vertex_faces[:, i], face_normals)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)

    return vertex_normals


def coords_to_theta_phi(vectors):

    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    phi   = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1, 1))

    return theta, phi
