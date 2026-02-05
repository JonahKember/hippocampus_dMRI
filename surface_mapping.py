import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.svm import SVC
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, binary_dilation



def create_partial_volume_mask(config, param_names=['DT_ADC','DT_AD','DT_RD','DT_FA','DKT_MK']):

    subject = config['subject']

    for hemi in ['L','R']:

        param_paths = [f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-{param}.nii.gz' for param in param_names]
        mask_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-mask.nii.gz'
        pvol_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-partial_volume.nii.gz'


        # Get mask.
        mask_nii = nib.load(mask_path)
        mask     = mask_nii.get_fdata() > 0
        n_vox    = mask.shape[0]

        # Load parameter data.
        n_param = len(param_names)
        params = np.zeros([n_vox, n_vox, n_vox, n_param])
        for idx, param_path in enumerate(param_paths):

            param_nii = nib.load(param_path)
            params[:,:,:,idx] = param_nii.get_fdata()


        # Erode and dilate mask.
        params_masked = params.copy()
        params_masked[mask == False] = 0

        mask_eroded  = binary_erosion(mask, iterations=3)
        mask_dilated = binary_dilation(mask, iterations=3)

        mask_flat         = mask.flatten()
        mask_eroded_flat  = mask_eroded.flatten()
        mask_dilated_flat = mask_dilated.flatten()


        outside_flat    = (~mask_flat) & mask_dilated_flat
        candidates_flat = mask_flat & (~mask_eroded_flat)


        # Create a voxel by parameter dataframe for classification.
        params_df = pd.DataFrame()

        for param_idx in range(n_param):
            params_df[param_names[param_idx]] = params[:,:,:,param_idx].flatten()

        # Assign voxels as intra (mask + eroded), extra (dilated) or unknown (mask - eroded).
        params_df['voxel_group'] = 'NA'
        params_df.loc[mask_eroded_flat, 'voxel_group'] = 'intra_hipp'
        params_df.loc[outside_flat,     'voxel_group'] = 'extra_hipp'
        params_df.loc[candidates_flat,  'voxel_group'] = 'unknown'

        params_df = params_df[params_df['voxel_group'] != 'NA']


        # Divide data into train/test.
        df_train = params_df[params_df['voxel_group'].isin(['intra_hipp','extra_hipp'])]
        df_test  = params_df[params_df['voxel_group'].isin(['unknown'])]

        X_train = df_train[param_names].to_numpy()
        X_test = df_test[param_names].to_numpy()

        y_train = df_train['voxel_group']


        # Fit voxel classifier on diffusion data predicting intra-hipp versus extra-hipp.
        svc = SVC(kernel='linear', probability=True).fit(X_train, y_train)

        params_df.loc[df_train.index,'voxel_pred'] = svc.predict(X_train)
        params_df.loc[df_test.index,'voxel_pred'] = svc.predict(X_test)

        intra_idx = list(svc.classes_).index('intra_hipp')
        params_df.loc[df_train.index,'prob_intra'] = svc.predict_proba(X_train)[:,intra_idx]
        params_df.loc[df_test.index,'prob_intra'] = svc.predict_proba(X_test)[:,intra_idx]

        params_df.to_csv(f'output/partial_volume_data__hemi-{hemi}.csv', index=False)


        # Create partial volume mask with probability of intra-hippocampl.
        mask_params = params_df[params_df['voxel_group'].isin(['intra_hipp','unknown'])]
        mask_prob_intra = mask_params['prob_intra'].values

        prob_intra_flat = np.zeros(mask.shape).flatten()
        prob_intra_flat[mask_flat] = mask_prob_intra
        prob_intra_flat[mask_eroded_flat] = 1

        prob_intra = prob_intra_flat.reshape(mask.shape)

        # Write partial volume to NIFTI.
        prob_intra_nii = nib.Nifti1Image(
            prob_intra,
            affine=mask_nii.affine,
            header=mask_nii.header
        )

        nib.save(prob_intra_nii, pvol_path)




def run_surface_mapping(config, param_names=['DT_ADC','DT_AD','DT_RD','DT_FA','DKT_MK'], surface_type='inner'):

    subject = config['subject']

    for hemi in ['L','R']:

        surf_path  = f'output/sub-{subject}_hemi-{hemi}_space-B0_den-0p5mm_label-hipp_{surface_type}.surf.gii'
        pvol_path  = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-partial_volume.nii.gz'

        for param_name in param_names:

            vol_path   = f'output/sub-{subject}_hemi-{hemi}_space-cropB0_desc-{param_name}.nii.gz'

            vertex_vals = _map_to_surface(vol_path, surf_path, pvol_path)
            vertex_gii  = _create_darray_GIFTI(vertex_vals, hemi, map_name=param_name)

            nib.save(vertex_gii, f'output/sub-{subject}_hemi-{hemi}_desc-{param_name}.{hemi}.func.gii')



def _map_to_surface(vol_path, surf_path, pvol_path, radius_mm=5, sigma_mm=0.5):
    ''' Perform volume-to-surface mapping of parameter data to hippocampal surface.

        To reduce partial volume effects, values are weighted by a probablistic intra-hippocampal mask.

        Parameters
        ----------
        vol_path:  Path to NIFTI of parameter map.
        surf_path: Path to GIFTI (.surf.gii) surface for mapping.
        pvol_path: Path to NIFTI with partial-volume data (0-1: prob. intra-hippocampal).
        radius_mm: Radius of search box when considering voxels for mapping.
        sigma_mm:  Sigma of the Gaussian used in distance-weighting.

        Returns
        -------
        vertex_values
        '''


    # Load data data.
    mask_nii = nib.load(pvol_path)
    mask_data = mask_nii.get_fdata()
    mask_data_flat = mask_data[mask_data > 0]

    vol_nii    = nib.load(vol_path)
    vol_data   = vol_nii.get_fdata()
    vol_affine = vol_nii.affine

    surf       = nib.load(surf_path)
    vertex_mm  = surf.darrays[1].data
    n_vertices = len(vertex_mm)


    # Get coordinates of masked voxels.
    voxel_indices = np.column_stack(np.where(mask_data > 0))
    voxel_mm = nib.affines.apply_affine(vol_affine, voxel_indices)

    # Get masked-voxel data.
    n_voxels = len(voxel_indices)
    voxel_data = np.full(n_voxels, np.nan)

    for idx, voxel_idx in enumerate(voxel_indices):
        x, y, z = voxel_idx
        voxel_data[idx] = vol_data[x, y, z]


    # Map voxel data to surface-vertex, weighted by distance and partial-volume.
    voxel_tree = cKDTree(voxel_mm)

    vertex_values = np.full(n_vertices, np.nan)
    for v_idx, vertex in enumerate(vertex_mm):

        nearby_vox = voxel_tree.query_ball_point(vertex, r=radius_mm)

        nearby_vox_coords = voxel_mm[nearby_vox]
        distances = np.linalg.norm(nearby_vox_coords - vertex, axis=1)

        # Voxels weighted by partial-volume and gaussian-weighted distance.
        distance_W = np.exp(-0.5 * (distances / sigma_mm)**2)
        partial_volume_W = mask_data_flat[nearby_vox]
        W = distance_W * partial_volume_W

        vertex_values[v_idx] = np.average(voxel_data[nearby_vox], weights=W)

    return vertex_values



def _create_darray_GIFTI(data, hemi, map_name='None'):

    intent = nib.nifti1.intent_codes['NIFTI_INTENT_NONE']

    darray = nib.gifti.GiftiDataArray(np.array(data, dtype='float32'), intent=intent)
    darray.meta = nib.gifti.GiftiMetaData({'Name':map_name})

    if hemi == 'L': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusLeft'})
    if hemi == 'R': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'HippocampusRight'})

    gii = nib.GiftiImage(darrays=[darray], meta=meta)

    return gii
