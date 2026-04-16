from hippocampus_dMRI_T1_space import io_utils
import numpy as np
import nibabel as nib

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel


def fit_noddi(config, lambda_par=1.1e-9, lambda_iso=3.0e-9):

    for hemi in ['L','R']:

        paths = io_utils.get_paths(config, hemi)

        bvals_path  = paths['bvals']
        bvecs_path  = paths['bvecs']
        dwi_path    = paths['dwi_upsampled']

        ndi_output = paths['noddi_NDI']
        odi_output = paths['noddi_ODI']
        mu_output  = paths['noddi_mu']

        # Prepare data.
        dwi_nii  = nib.load(dwi_path)
        data = dwi_nii.get_fdata()

        # Prepare acquisition scheme.
        bvals = np.loadtxt(bvals_path)
        bvecs = np.loadtxt(bvecs_path)

        bvals = bvals * 1e6 # Convert to s/m2
        bvecs = bvecs.T
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

        # Set up NODDI model.
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()

        bundle = SD1WatsonDistributed(models=[stick, zeppelin])

        bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', lambda_par)

        noddi = MultiCompartmentModel(models=[ball, bundle])
        noddi.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)

        # Fit model.
        noddi_fit = noddi.fit(scheme, data)

        # Write estimated parameters to NIFTI.
        ODI = noddi_fit.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
        NDI = (
            noddi_fit.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
            noddi_fit.fitted_parameters['partial_volume_1']
        )
        mu = noddi_fit.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_mu']


        nib.save(nib.Nifti1Image(NDI, dwi_nii.affine), ndi_output)
        nib.save(nib.Nifti1Image(ODI, dwi_nii.affine), odi_output)
        nib.save(nib.Nifti1Image(mu, dwi_nii.affine),  mu_output)
