import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


hipp_vol  = 'output/sub-HCD0001305_hemi-R_space-cropB0_desc-DT_FA.nii.gz'
hipp_surf = 'output/sub-HCD0001305_hemi-R_space-B0_den-0p5mm_label-hipp_midthickness.surf.gii'

# Load volume
nii = nib.load(hipp_vol)
data = nii.get_fdata()
affine = nii.affine

# Load surface
gii = nib.load(hipp_surf)
coords = gii.darrays[1].data
coords_vox = nib.affines.apply_affine(np.linalg.inv(affine), coords)


fig, ax = plt.subplots(2, 3, figsize=(12,8))
ax = ax.flatten()

for idx, slice_idx in enumerate(np.linspace(30,65,6)):


    # Plot volume data
    slice_idx = int(slice_idx)
    ax[idx].imshow(data[slice_idx, :, :].T, cmap='gray', origin='lower')

    # Plot surface vertices.
    mask = np.abs(coords_vox[:,0] - slice_idx) < .5
    ax[idx].scatter(
        coords_vox[mask,1],
        coords_vox[mask,2],
        s=1, c='yellow'
    )
    ax[idx].axis('off')

plt.tight_layout()
plt.show()