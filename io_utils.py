import os
import json
from pathlib import Path
from dataclasses import dataclass


os.makedirs(f'output', exist_ok=True)
os.makedirs(f'output/surf', exist_ok=True)
os.makedirs(f'output/anat', exist_ok=True)
os.makedirs(f'output/dwi', exist_ok=True)
os.makedirs(f'output/params', exist_ok=True)


def get_paths(config, hemi):

    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    subject   = config['subject']
    input_dir = config['input']

    paths = {
        'dwi':          f'{input_dir}/sub-{subject}_dwi.nii.gz',
        'bvals':        f'{input_dir}/bvals',
        'bvecs':        f'{input_dir}/bvecs',

        'B0_to_T1':     f'{input_dir}/b0_to_T1.txt',

        'T1':           f'{input_dir}/sub-{subject}_hemi-{hemi}_space-cropT1w_desc-preproc_T1w.nii.gz',
        'subfields':    f'{input_dir}/sub-{subject}_hemi-{hemi}_space-cropT1w_desc-subfields_atlas-multihist7_dseg.nii.gz',
        'midthickness': f'{input_dir}/sub-{subject}_hemi-{hemi}_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii',
        'inner':        f'{input_dir}/sub-{subject}_hemi-{hemi}_space-T1w_den-0p5mm_label-hipp_inner.surf.gii',
        'outer':        f'{input_dir}/sub-{subject}_hemi-{hemi}_space-T1w_den-0p5mm_label-hipp_outer.surf.gii',

        'T1_space-B0':           f'output/anat/sub-{subject}_hemi-{hemi}_desc-preproc_T1w.nii.gz',
        'subfields_space-B0':    f'output/anat/sub-{subject}_hemi-{hemi}_desc-subfields.nii.gz',

        'midthickness_space-B0': f'output/surf/sub-{subject}_hemi-{hemi}_label-hipp_midthickness.surf.gii',
        'inner_space-B0':        f'output/surf/sub-{subject}_hemi-{hemi}_label-hipp_inner.surf.gii',
        'outer_space-B0':        f'output/surf/sub-{subject}_hemi-{hemi}_label-hipp_outer.surf.gii',

        'dwi_space-B0':          f'output/dwi/sub-{subject}_hemi-{hemi}_upsampled_dwi.nii.gz',
        'mask':                  f'output/anat/sub-{subject}_hemi-{hemi}_mask.nii.gz',
        'distance':              f'output/anat/sub-{subject}_hemi-{hemi}_outer-inner-distance.nii.gz',
        'tissue_seg':            f'output/anat/sub-{subject}_hemi-{hemi}_tissue_seg.nii.gz',
        'pvol':                  f'output/anat/sub-{subject}_hemi-{hemi}_tissue_pvol.nii.gz',


        'diffusion_tensor':          f'output/params/sub-{subject}_hemi-{hemi}_DT_tensor.nii.gz',
        'diffusion_kurtosis_tensor': f'output/params/sub-{subject}_hemi-{hemi}_DKT_tensor.nii.gz',

        'DT_adc':         f'output/params/sub-{subject}_hemi-{hemi}_DT_ADC.nii.gz',
        'DT_fa':          f'output/params/sub-{subject}_hemi-{hemi}_DT_FA.nii.gz',
        'DT_ad':          f'output/params/sub-{subject}_hemi-{hemi}_DT_AD.nii.gz',
        'DT_rd':          f'output/params/sub-{subject}_hemi-{hemi}_DT_RD.nii.gz',

        'DT_eigenvals':      f'output/params/sub-{subject}_hemi-{hemi}_DT_eigenvals.nii.gz',
        'DT_eigenvecs':      f'output/params/sub-{subject}_hemi-{hemi}_DT_eigenvecs.nii.gz',

        'DKT_params':     f'output/params/sub-{subject}_hemi-{hemi}_DKT_params.nii.gz',
        'DKT_MK':         f'output/params/sub-{subject}_hemi-{hemi}_DKT_MK.nii.gz',
        'DKT_AK':          f'output/params/sub-{subject}_hemi-{hemi}_DKT_AK.nii.gz',
        'DKT_RK':          f'output/params/sub-{subject}_hemi-{hemi}_DKT_RK.nii.gz',
        'DKT_KFA':         f'output/params/sub-{subject}_hemi-{hemi}_DKT_FKA.nii.gz',

        'diffusion_normal':       f'output/params/sub-{subject}_hemi-{hemi}_diffusion_normal.nii.gz',
        'diffusion_tangential':   f'output/params/sub-{subject}_hemi-{hemi}_diffusion_tangential.nii.gz',
        'neurite_volume_fraction': f'output/params/sub-{subject}_hemi-{hemi}_neurite_volume_fraction.nii.gz',
        'diffusion_sphere':        f'output/params/sub-{subject}_hemi-{hemi}_diffusion_sphere.nii.gz',
        'kurtosis_sphere':         f'output/params/sub-{subject}_hemi-{hemi}_kurtosis_sphere.nii.gz',
    }

    return paths
