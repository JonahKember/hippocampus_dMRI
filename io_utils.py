import os
import json



def get_paths(config, hemi):

    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    output = config['output']
    os.makedirs(f'{output}', exist_ok=True)
    os.makedirs(f'{output}/surf', exist_ok=True)
    os.makedirs(f'{output}/anat', exist_ok=True)
    os.makedirs(f'{output}/dwi', exist_ok=True)
    os.makedirs(f'{output}/params', exist_ok=True)

    subject   = config['subject']

    paths = {
        'subject':            subject,
        'bvals':              config['bvals'],
        'bvecs':              config['bvecs'],
        'dwi':                config['dwi'],

        'T1_input':           config[f'hipp_crop_T1_{hemi}'],
        'subfields_input':    config[f'hipp_volume_{hemi}'],
        'midthickness_input': config[f'hipp_midthickness_{hemi}'],
        'inner_input':        config[f'hipp_inner_{hemi}'],
        'outer_input':        config[f'hipp_outer_{hemi}'],

        'T1':                    f'{output}/anat/sub-{subject}_hemi-{hemi}_T1w.nii.gz',
        'subfields':             f'{output}/anat/sub-{subject}_hemi-{hemi}_subfields.nii.gz',
        'midthickness':          f'{output}/surf/sub-{subject}_hemi-{hemi}_midthickness.surf.gii',
        'inner':                 f'{output}/surf/sub-{subject}_hemi-{hemi}_inner.surf.gii',
        'outer':                 f'{output}/surf/sub-{subject}_hemi-{hemi}_outer.surf.gii',
        'registration':          f'{output}/anat/sub-{subject}_hemi-{hemi}_from-orig_to-refined',


        'layers':                f'{output}/anat/sub-{subject}_hemi-{hemi}_layers_mask.nii.gz',
        'mask':                  f'{output}/anat/sub-{subject}_hemi-{hemi}_mask.nii.gz',
        'distance':              f'{output}/anat/sub-{subject}_hemi-{hemi}_distance.nii.gz',
        'tissue_seg':            f'{output}/anat/sub-{subject}_hemi-{hemi}_tissue_seg_refined.nii.gz',
        'pvol':                  f'{output}/anat/sub-{subject}_hemi-{hemi}_pvol_refined.nii.gz',

        'T1_refined':            f'{output}/anat/sub-{subject}_hemi-{hemi}_T1w_refined.nii.gz',
        'layers_refined':        f'{output}/anat/sub-{subject}_hemi-{hemi}_layers_mask_refined.nii.gz',
        'mask_refined':          f'{output}/anat/sub-{subject}_hemi-{hemi}_mask_refined.nii.gz',
        'distance_refined':      f'{output}/anat/sub-{subject}_hemi-{hemi}_distance_refined.nii.gz',

        'midthickness_refined':  f'{output}/surf/sub-{subject}_hemi-{hemi}_midthickness_refined.surf.gii',
        'inner_refined':         f'{output}/surf/sub-{subject}_hemi-{hemi}_inner_refineds.surf.gii',
        'outer_refined':         f'{output}/surf/sub-{subject}_hemi-{hemi}_outer_refined.surf.gii',


        'dwi_upsampled':         f'{output}/dwi/sub-{subject}_hemi-{hemi}_upsampled_dwi.nii.gz',

        'DT_tensor':             f'{output}/params/sub-{subject}_hemi-{hemi}_DT_tensor.nii.gz',
        'DKT_tensor':            f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_tensor.nii.gz',

        'mean_B0':                f'{output}/params/sub-{subject}_hemi-{hemi}_mean_B0.nii.gz',
        'mean_B0_smooth':         f'{output}/params/sub-{subject}_hemi-{hemi}_mean_B0_smooth.nii.gz',
        'DT_adc':                 f'{output}/params/sub-{subject}_hemi-{hemi}_DT_ADC.nii.gz',
        'DT_fa':                  f'{output}/params/sub-{subject}_hemi-{hemi}_DT_FA.nii.gz',
        'DT_ad':                  f'{output}/params/sub-{subject}_hemi-{hemi}_DT_AD.nii.gz',
        'DT_rd':                  f'{output}/params/sub-{subject}_hemi-{hemi}_DT_RD.nii.gz',

        'DT_eigenvals':           f'{output}/params/sub-{subject}_hemi-{hemi}_DT_eigenvals.nii.gz',
        'DT_eigenvecs':           f'{output}/params/sub-{subject}_hemi-{hemi}_DT_eigenvecs.nii.gz',

        'DKT_params':             f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_params.nii.gz',
        'DKT_MK':                 f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_MK.nii.gz',
        'DKT_AK':                 f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_AK.nii.gz',
        'DKT_RK':                 f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_RK.nii.gz',
        'DKT_KFA':                f'{output}/params/sub-{subject}_hemi-{hemi}_DKT_FKA.nii.gz',

        'diffusion_normal':       f'{output}/params/sub-{subject}_hemi-{hemi}_diffusion_normal.nii.gz',
        'diffusion_tangential':   f'{output}/params/sub-{subject}_hemi-{hemi}_diffusion_tangential.nii.gz',
        'diffusion_sphere':       f'{output}/params/sub-{subject}_hemi-{hemi}_diffusion_sphere.nii.gz',
        'kurtosis_sphere':        f'{output}/params/sub-{subject}_hemi-{hemi}_kurtosis_sphere.nii.gz',
        'diffusion_log_ratio':    f'{output}/params/sub-{subject}_hemi-{hemi}_diffusion_log_ratio.nii.gz',
        'spatial_gradient':       f'{output}/params/sub-{subject}_hemi-{hemi}_spatial_gradient.nii.gz',

        'noddi_NDI':              f'{output}/params/sub-{subject}_hemi-{hemi}_noddi_NDI.nii.gz',
        'noddi_ODI':              f'{output}/params/sub-{subject}_hemi-{hemi}_noddi_ODI.nii.gz',
        'noddi_mu':               f'{output}/params/sub-{subject}_hemi-{hemi}_noddi_mu.nii.gz',
    }

    for input in ['T1','subfields','midthickness','inner','outer']:
        if not os.path.exists(paths[input]):
            os.system(f'cp {paths[f"{input}_input"]} {paths[input]}')

    return paths
